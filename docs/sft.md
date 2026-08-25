# sft — teacher-generated instruction data (phase-6 §6)

Pipeline: corpus records → 5 task types ke seeds → teacher → validators +
filters → chat-template JSONL. Poora pipeline OFFLINE testable hai
(FakeTeacher); REAL generation sirf teacher key ke saath.

## Status (M3)

- ✅ Pipeline code + offline dry-run (`--dry-run`)
- ⛔ Real generation GATED: owner se `RYTH_TEACHER_API_KEY` +
  `RYTH_TEACHER_MODEL` chahiye (proxy base-url bhi tab)

## Offline smoke (aaj chalega)

```bash
python3 -m sft.cli generate --src <koi-chhota-.py-dir> \
    --out /tmp/sft_smoke.jsonl --dry-run
```

## Real run (key aane par)

```bash
export RYTH_TEACHER_API_KEY=sk-...        # owner proxy key
export RYTH_TEACHER_MODEL=<backend-name>  # Nemotron-class
ryth-sft generate --src corpus_out --target 5000 \
  --out data/sft_v1.jsonl --tokenizer tok/tokenizer.json \
  --base-url https://<proxy-endpoint>/v1   # agar direct OpenRouter nahi
```

Acceptance (spec §6): ~5–10k examples, pass_rate ≥ 0.90 (stats JSON me;
CLI warn karta hai agar neeche), owner spot-check sample.

Run behavior jo API-spend/resilience ke liye matter karta hai:

- `--target N` = jitne examples **PASS** hote hi teacher calls band
  (spend cap — poora seed-pool jalata nahi).
- Har PASS row turant flush hoti hai + `<out>.stats.json` sidecar —
  mid-run crash par bhi likha hua data bacha rehta hai (sidecar me
  `"crashed": true` + `n_written`).
- Missing key/model jaisi SETUP galti clean SystemExit hai (traceback
  nahi); per-seed transient teacher errors soft hain — reason stats ke
  `filter_reasons.teacher_error` me ginta hai.
- Dedup case-preserving hai (`Add` ≠ `add`) — sirf whitespace
  normalize hota hai.

## Output format

Har row: `task`, `messages` (persona-system/user/assistant), `text`
(W2 chat-template rendered), optional `token_ids` (sirf --tokenizer ke
saath — provisional tokenizer bake karne se bacho), `meta`. Trainer
`text`/`token_ids` seedha consume karega.

Teacher meta-directives stored turns me leak NAHI hote (tested) — persona
`SYSTEM_PROMPT` hi dataset me jaata hai.

## Security

⚠ `test_gen` validator generated asserts LOCAL subprocess me execute karta
hai (timeout 10s, no network jail) — trusted machine par hi chalao.
