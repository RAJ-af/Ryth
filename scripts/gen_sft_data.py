#!/usr/bin/env python3
"""Standard-library tool to generate SFT chat training data using LLM APIs.

Supports Gemini and OpenAI-compatible providers via environment variables:
  PROVIDER=gemini|openai_compat
  API_KEY=<key>  (or GEMINI_API_KEY / OPENAI_API_KEY)
  MODEL=<model_name>
  BASE_URL=<base_url>  (optional for openai_compat, default https://openrouter.ai/api/v1)

Usage:
  python3 scripts/gen_sft_data.py --lang en --target 100
  python3 scripts/gen_sft_data.py data/sft/topics_starter.txt --lang hi --target 50 --rpm 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

PROHIBITED_NAMES = [
    "openai", "google", "anthropic", "meta", "chatgpt", "gpt-3", "gpt-4",
    "gpt-4o", "gemini", "llama", "claude", "mistral", "deepseek", "nemotron"
]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def validate_script_and_content(q: str, a: str, lang: str) -> tuple[bool, str]:
    """Validate char limit, script rules, and persona requirements for pair."""
    if not isinstance(q, str) or not isinstance(a, str):
        return False, "non-string content"

    q_strip = q.strip()
    a_strip = a.strip()

    if not q_strip or not a_strip:
        return False, "empty content"

    if len(q_strip) > 1500 or len(a_strip) > 1500:
        return False, f"exceeds 1500 chars limit (q={len(q_strip)}, a={len(a_strip)})"

    # Script check
    q_has_dev = bool(DEVANAGARI_RE.search(q_strip))
    a_has_dev = bool(DEVANAGARI_RE.search(a_strip))

    if lang == "hi":
        if not (q_has_dev or a_has_dev):
            return False, "hi requires Devanagari script"
    else:  # en or hinglish
        if q_has_dev or a_has_dev:
            return False, f"{lang} must NOT contain Devanagari script"

    # Persona check: no competitor names
    combined = (q_strip + " " + a_strip).lower()
    for name in PROHIBITED_NAMES:
        if name in combined:
            return False, f"mentions prohibited AI model/company name: {name}"

    return True, ""


def build_prompt(topics: list[str], lang: str, count: int) -> str:
    lang_rules = {
        "en": "Write all questions and answers in simple, clear English.",
        "hi": "Write all questions and answers in Devanagari Hindi script ONLY (हिंदी). Do NOT use English/Latin alphabet.",
        "hinglish": "Write all questions and answers in Hinglish (Hindi written using Roman/Latin alphabet mixed with English). NEVER use Devanagari script.",
    }.get(lang, "Write all questions and answers in clear English.")

    topics_list = "\n".join(f"- {t}" for t in topics)

    prompt = f"""You are generating synthetic instruction training data for an AI assistant named "Ryth".

Generate exactly {count} distinct user-assistant conversation pairs based on these topics:
{topics_list}

Language Rule:
{lang_rules}

Persona Rules:
- The assistant is called "Ryth". NEVER mention any AI company (OpenAI, Google, Anthropic, Meta, etc.) or model names (GPT, Gemini, Llama, Claude, etc.).
- Assistant answers should be 2 to 6 sentences long, clear, and accurate.
- Include small code snippets or examples when relevant.
- If asking about unknown or unverified facts, have the assistant say "I'm not sure".

Output Format:
Return a valid JSON array of objects. Each object must have "user" and "assistant" string keys.
Example:
[
  {{"user": "Question here", "assistant": "Answer here"}},
  ...
]

Return ONLY the raw JSON array (or wrapped in ```json ... ```). Do not include any other text outside JSON.
"""
    return prompt


def parse_llm_json_response(raw_text: str) -> list[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Extract bracketed JSON list
    match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def call_llm_api(
    provider: str,
    prompt: str,
    api_key: str,
    model: str,
    base_url: str | None = None,
    transport=None,
    attempts: int = 4,
    backoff_s: float = 2.0,
    sleep_fn=time.sleep,
) -> str:
    """Send HTTP request to specified LLM API provider using urllib."""
    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
        }
        headers = {"Content-Type": "application/json"}
    elif provider == "openai_compat":
        burl = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        url = f"{burl}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")

    def _default_transport(u, p, h):
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(u, data=data, headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode("utf-8", "replace")

    tp = transport or _default_transport
    last_err = ""

    for attempt in range(attempts):
        status, body = tp(url, payload, headers)
        if status == 200:
            try:
                res_json = json.loads(body)
                if provider == "gemini":
                    return res_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return res_json["choices"][0]["message"]["content"]
            except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
                raise RuntimeError(f"Unexpected API response shape ({e}): {body[:200]}")

        last_err = f"HTTP {status}: {body[:200]}"
        if status != 429 and status < 500:
            break  # Fast-fail on non-retryable 4xx

        sleep_fn(backoff_s * (2 ** attempt))

    raise RuntimeError(f"API call failed after {attempts} attempts: {last_err}")


def normalize_question_hash(q: str) -> str:
    norm = unicodedata.normalize("NFC", q.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def load_existing_hashes_and_count(out_path: str) -> tuple[set[str], int]:
    seen = set()
    count = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msgs = obj.get("messages", [])
                    user_msg = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
                    if user_msg:
                        seen.add(normalize_question_hash(user_msg))
                        count += 1
                except json.JSONDecodeError:
                    continue
    return seen, count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate SFT training data via LLM API.")
    parser.add_argument("topics_file", nargs="?", default="data/sft/topics_starter.txt",
                        help="Topics text file (one topic per line)")
    parser.add_argument("--lang", choices=["en", "hi", "hinglish"], default="en",
                        help="Target language (en, hi, hinglish)")
    parser.add_argument("--pairs_per_call", type=int, default=20,
                        help="Number of pairs to request per API call")
    parser.add_argument("--target", type=int, default=100,
                        help="Target number of valid pairs to generate")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path (defaults to data/sft/<lang>.jsonl)")
    parser.add_argument("--rpm", type=float, default=10.0,
                        help="Requests per minute rate limit")

    args = parser.parse_args(argv)

    out_file = args.out or f"data/sft/{args.lang}.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    # Provider configuration from environment
    provider = os.environ.get("PROVIDER", "openai_compat").lower()
    api_key = (
        os.environ.get("API_KEY") or
        os.environ.get("GEMINI_API_KEY") or
        os.environ.get("OPENAI_API_KEY") or
        ""
    )
    model = os.environ.get("MODEL", "google/gemini-2.0-flash-001" if provider == "openai_compat" else "gemini-2.0-flash")
    base_url = os.environ.get("BASE_URL")

    # Load topics
    if not os.path.exists(args.topics_file):
        print(f"[gen_sft_data] Topics file not found: {args.topics_file}", file=sys.stderr)
        return 1

    with open(args.topics_file, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not topics:
        print(f"[gen_sft_data] No topics found in {args.topics_file}", file=sys.stderr)
        return 1

    # Load existing hashes and count for resume capability
    seen_hashes, existing_count = load_existing_hashes_and_count(out_file)
    print(f"[gen_sft_data] Lang: {args.lang} | Existing in {out_file}: {existing_count} | Target: {args.target}")

    if existing_count >= args.target:
        print(f"[gen_sft_data] Target ({args.target}) already reached. Nothing to generate.")
        return 0

    collected = existing_count
    topic_idx = 0
    out_fp = open(out_file, "a", encoding="utf-8")

    interval_s = 60.0 / max(0.1, args.rpm)

    try:
        while collected < args.target:
            t_start = time.time()
            batch_topics = [topics[(topic_idx + i) % len(topics)] for i in range(min(5, len(topics)))]
            topic_idx = (topic_idx + 5) % len(topics)

            needed = min(args.pairs_per_call, args.target - collected)
            prompt = build_prompt(batch_topics, args.lang, needed)

            print(f"[gen_sft_data] Requesting {needed} pairs (collected {collected}/{args.target})...")

            try:
                raw_resp = call_llm_api(provider, prompt, api_key, model, base_url=base_url)
            except Exception as e:
                print(f"[gen_sft_data] API call error: {e}", file=sys.stderr)
                time.sleep(5.0)
                continue

            pairs = parse_llm_json_response(raw_resp)
            if not pairs:
                print("[gen_sft_data] Failed to parse JSON array from response", file=sys.stderr)
                continue

            added = 0
            for item in pairs:
                if not isinstance(item, dict):
                    continue
                q = item.get("user") or item.get("question") or ""
                a = item.get("assistant") or item.get("answer") or ""

                ok, reason = validate_script_and_content(q, a, args.lang)
                if not ok:
                    continue

                q_hash = normalize_question_hash(q)
                if q_hash in seen_hashes:
                    continue

                seen_hashes.add(q_hash)
                row = {
                    "messages": [
                        {"role": "user", "content": q.strip()},
                        {"role": "assistant", "content": a.strip()}
                    ]
                }
                out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fp.flush()
                collected += 1
                added += 1
                if collected >= args.target:
                    break

            print(f"[gen_sft_data] Added {added} valid pairs. Progress: {collected}/{args.target}")

            elapsed = time.time() - t_start
            sleep_time = max(0.0, interval_s - elapsed)
            if sleep_time > 0 and collected < args.target:
                time.sleep(sleep_time)

    finally:
        out_fp.close()

    print(f"[gen_sft_data] Complete: Saved {collected} total pairs to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
