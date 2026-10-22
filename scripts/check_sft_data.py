#!/usr/bin/env python3
"""Validate, summarize, and merge curated Ryth SFT JSONL datasets.

Run from the repository root:
  python scripts/check_sft_data.py check data/sft/raw
  python scripts/check_sft_data.py stats data/sft/raw
  python scripts/check_sft_data.py merge data/sft/raw data/sft/merged
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

LANGS = {"en", "hi", "hinglish"}
AI_TERMS = re.compile(
    r"\b(?:openai|chatgpt|gpt(?:[- ]?\d+(?:\.\d+)*)?|codex|google|gemini|"
    r"anthropic|claude|meta|llama|qwen|deepseek|nvidia|groq|mistral)\b",
    re.IGNORECASE,
)
PLACEHOLDERS = re.compile(r"\b(?:todo|lorem(?:\s+ipsum)?|tbd|placeholder)\b", re.I)
THINK_TAGS = re.compile(r"</?\s*(?:think|analysis|reasoning)\s*>", re.I)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)

def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", unicodedata.normalize("NFC", text).casefold())

def _trigrams(text: str) -> set[str]:
    words = _words(text)
    if len(words) < 3:
        return set()
    return {" ".join(words[i:i+3]) for i in range(len(words)-2)}

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _load_records(root: Path) -> tuple[list[dict[str, Any]], list[str], dict[str,int]]:
    records, errors, file_lines = [], [], {}
    files = sorted(root.glob("*.jsonl"))
    if not root.exists():
        return [], [f"{root}: directory does not exist"], {}
    for path in files:
        count = 0
        try:
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        errors.append(f"{path}:{line_no}: blank line is not a JSONL record")
                        continue
                    count += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"{path}:{line_no}: invalid JSON ({e.msg})")
                        continue
                    if not isinstance(obj, dict):
                        errors.append(f"{path}:{line_no}: record must be a JSON object")
                        continue
                    records.append({"path":path, "line":line_no, "obj":obj})
        except (OSError, UnicodeError) as e:
            errors.append(f"{path}: cannot read file ({e})")
        file_lines[str(path)] = count
        if count != 100:
            print(f"WARNING: {path}: contains {count} records (expected 100)", file=sys.stderr)
    return records, errors, file_lines

def _validate_record(item: dict[str, Any]) -> list[str]:
    obj, path, line = item["obj"], item["path"], item["line"]
    where = f"{path}:{line}"
    errors = []
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        return [f"{where}: messages must contain exactly [user, assistant]"]
    for i, role in enumerate(("user", "assistant")):
        msg = msgs[i]
        if not isinstance(msg, dict) or msg.get("role") != role or not isinstance(msg.get("content"), str):
            errors.append(f"{where}: messages[{i}] must have role={role!r} and string content")
    if errors:
        return errors
    q, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
    if not q: errors.append(f"{where}: user question is empty")
    if not a: errors.append(f"{where}: assistant answer is empty")
    if len(q) > 300: errors.append(f"{where}: question exceeds 300 characters")
    if len(a) > 600: errors.append(f"{where}: answer exceeds 600 characters")
    lang = obj.get("lang")
    if lang not in LANGS: errors.append(f"{where}: lang must be one of en, hi, hinglish")
    if not isinstance(obj.get("provider"), str) or not obj["provider"].strip():
        errors.append(f"{where}: provider must be a nonempty string")
    for field in ("topic", "angle"):
        if not isinstance(obj.get(field), str) or not obj[field].strip():
            errors.append(f"{where}: {field} must be a nonempty string")
    b = obj.get("batch")
    if isinstance(b, bool) or not (isinstance(b, int) or (isinstance(b, str) and b.strip())):
        errors.append(f"{where}: batch must be an integer or a nonempty string")
    for label, text in (("question", q), ("answer", a)):
        if THINK_TAGS.search(text): errors.append(f"{where}: {label} contains a think/reasoning tag")
        if PLACEHOLDERS.search(text) or "..." in text or "…" in text:
            errors.append(f"{where}: {label} contains placeholder/ellipsis text")
        if AI_TERMS.search(text): errors.append(f"{where}: {label} mentions a prohibited AI/company/model term")
    if lang == "hi":
        if not DEVANAGARI_RE.search(q): errors.append(f"{where}: Hindi question must contain Devanagari")
        if not DEVANAGARI_RE.search(a): errors.append(f"{where}: Hindi answer must contain Devanagari")
    elif lang in {"en", "hinglish"}:
        if DEVANAGARI_RE.search(q) or DEVANAGARI_RE.search(a):
            errors.append(f"{where}: {lang} text must not contain Devanagari")
    return errors

def check(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records, errors, _ = _load_records(root)
    for item in records:
        errors.extend(_validate_record(item))
    # Exact duplicates across all files: question alone AND answer alone.
    seen_q: dict[str, str] = {}
    seen_a: dict[str, str] = {}
    valid = []
    for item in records:
        msgs = item["obj"].get("messages", [])
        if len(msgs) != 2 or not all(isinstance(m,dict) and isinstance(m.get("content"),str) for m in msgs):
            continue
        where = f"{item['path']}:{item['line']}"
        kq, ka = _norm(msgs[0]["content"]), _norm(msgs[1]["content"])
        if kq and kq in seen_q:
            errors.append(f"{where}: exact duplicate question of {seen_q[kq]}")
        elif kq:
            seen_q[kq] = where
        if ka and ka in seen_a:
            errors.append(f"{where}: exact duplicate answer of {seen_a[ka]}")
        elif ka:
            seen_a[ka] = where
        valid.append(item)
    # Inverted trigram index: only compare questions sharing >=1 trigram.
    qsets = [(item, _trigrams(item["obj"]["messages"][0]["content"])) for item in valid
             if isinstance(item["obj"].get("messages"), list) and len(item["obj"]["messages"]) == 2
             and isinstance(item["obj"]["messages"][0], dict)
             and isinstance(item["obj"]["messages"][0].get("content"), str)]
    index: dict[str, list[int]] = defaultdict(list)
    reported = set()
    for i, (item, grams) in enumerate(qsets):
        candidates = set()
        for gram in grams: candidates.update(index[gram])
        for j in sorted(candidates):
            other, other_grams = qsets[j]
            if _jaccard(grams, other_grams) >= 0.8:
                pair = (j, i)
                if pair not in reported:
                    errors.append(f"{item['path']}:{item['line']}: near-duplicate question (>=0.80 Jaccard) of {other['path']}:{other['line']}")
                    reported.add(pair)
        for gram in grams: index[gram].append(i)
    # Reject generic answer openings whose first-eight-word skeleton occurs 6+ times.
    skeleton_counts = Counter()
    skeleton_for = []
    for item in valid:
        msgs = item["obj"].get("messages", [])
        if len(msgs) == 2 and isinstance(msgs[1],dict) and isinstance(msgs[1].get("content"),str):
            skeleton = " ".join(_words(msgs[1]["content"])[:8])
            skeleton_for.append((item,skeleton))
            if skeleton: skeleton_counts[skeleton] += 1
    for item, skeleton in skeleton_for:
        if skeleton and skeleton_counts[skeleton] >= 6:
            errors.append(f"{item['path']}:{item['line']}: answer opening skeleton occurs {skeleton_counts[skeleton]} times")
    langs = Counter()
    for item in records:
        lang = item["obj"].get("lang")
        if lang in LANGS: langs[lang] += 1
    print("Totals by language: " + ", ".join(f"{k}={langs[k]}" for k in sorted(LANGS)))
    return records, errors

def stats(root: Path) -> int:
    records, errors = check(root)
    c_lang, c_provider, c_topic = Counter(), Counter(), Counter()
    for r in records:
        o=r["obj"]; c_lang[o.get("lang","?")]+=1
        c_provider[o.get("provider","?")]+=1; c_topic[o.get("topic","?")]+=1
    for title, counter in (("Language",c_lang),("Provider",c_provider),("Topic",c_topic)):
        print(f"\n{title}:")
        for key,count in sorted(counter.items()): print(f"  {key}: {count}")
    if errors:
        print("\nValidation errors:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0

def merge(raw: Path, out: Path, exclude: set[str]) -> int:
    records, errors = check(raw)
    if errors:
        print("Merge aborted: raw data has validation errors.", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    selected=[r for r in records if r["obj"].get("provider") not in exclude]
    payloads={lang:[] for lang in sorted(LANGS)}
    manifest={"total":0,"by_language":{},"by_provider":{},"by_batch":{}}
    for r in selected:
        o=r["obj"]; lang=o["lang"]
        payloads[lang].append(json.dumps({"messages":o["messages"]},ensure_ascii=False,separators=(",",":")))
        manifest["total"]+=1
        for key in ("lang","provider","batch"):
            bucket={"lang":"by_language","provider":"by_provider","batch":"by_batch"}[key]
            val=str(o[key]); manifest[bucket][val]=manifest[bucket].get(val,0)+1
    # Stable output order: preserve sorted source path/line traversal.
    out.mkdir(parents=True,exist_ok=True)
    staged={}
    for lang, lines in payloads.items():
        staged[out/f"{lang}.jsonl"]="\n".join(lines)+("\n" if lines else "")
    staged[out/"manifest.json"]=json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"
    # Write temp files first, then replace destination files.
    temps=[]
    try:
        for dest, content in staged.items():
            tmp=dest.with_name(dest.name+".tmp")
            tmp.write_text(content,encoding="utf-8")
            temps.append((tmp,dest))
        for tmp,dest in temps: tmp.replace(dest)
    except OSError as e:
        for tmp,_ in temps:
            try: tmp.unlink(missing_ok=True)
            except OSError: pass
        print(f"Merge failed while writing output: {e}",file=sys.stderr)
        return 1
    print(f"Merged {manifest['total']} records into {out}")
    return 0

def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("check"); c.add_argument("raw",type=Path)
    s=sub.add_parser("stats"); s.add_argument("raw",type=Path)
    m=sub.add_parser("merge"); m.add_argument("raw",type=Path); m.add_argument("out",type=Path)
    m.add_argument("--exclude-provider",action="append",default=[])
    a=p.parse_args(argv)
    if a.command=="check":
        _, errors=check(a.raw)
        if errors: print("\n".join(errors),file=sys.stderr); return 1
        print("Validation passed."); return 0
    if a.command=="stats": return stats(a.raw)
    return merge(a.raw,a.out,set(a.exclude_provider))
if __name__=="__main__":
    raise SystemExit(main())
