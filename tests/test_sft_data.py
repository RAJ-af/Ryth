"""Unit tests for SFT data generator (scripts/gen_sft_data.py)."""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import patch

from scripts.gen_sft_data import (
    build_prompt,
    call_llm_api,
    load_existing_hashes_and_count,
    normalize_question_hash,
    parse_llm_json_response,
    validate_script_and_content,
)


def test_validate_script_and_content_en():
    # Valid English
    ok, err = validate_script_and_content("What is Python?", "Python is a programming language.", "en")
    assert ok, err

    # Devanagari in English fails
    ok, err = validate_script_and_content("What is Python?", "Python एक भाषा है।", "en")
    assert not ok
    assert "Devanagari" in err


def test_validate_script_and_content_hi():
    # Valid Hindi Devanagari
    ok, err = validate_script_and_content("पाइथन क्या है?", "पाइथन एक प्रोग्रामिंग भाषा है।", "hi")
    assert ok, err

    # English-only in Hindi fails
    ok, err = validate_script_and_content("What is Python?", "Python is a programming language.", "hi")
    assert not ok
    assert "Devanagari" in err


def test_validate_script_and_content_hinglish():
    # Valid Hinglish
    ok, err = validate_script_and_content("Python kya hai?", "Python ek simple programming language hai.", "hinglish")
    assert ok, err

    # Devanagari in Hinglish fails
    ok, err = validate_script_and_content("Python kya hai?", "Python एक भाषा है।", "hinglish")
    assert not ok
    assert "Devanagari" in err


def test_validate_script_and_content_limits_and_persona():
    # Too long (>1500 chars)
    long_q = "a" * 1501
    ok, err = validate_script_and_content(long_q, "short answer", "en")
    assert not ok
    assert "1500" in err

    # Prohibited name
    ok, err = validate_script_and_content("Are you ChatGPT?", "No, I am Ryth created by OpenAI.", "en")
    assert not ok
    assert "prohibited" in err or "openai" in err.lower()


def test_parse_llm_json_response():
    # Plain JSON
    raw = '[{"user": "q1", "assistant": "a1"}]'
    res = parse_llm_json_response(raw)
    assert len(res) == 1 and res[0]["user"] == "q1"

    # Wrapped in ```json ... ```
    raw_md = '```json\n[{"user": "q2", "assistant": "a2"}]\n```'
    res_md = parse_llm_json_response(raw_md)
    assert len(res_md) == 1 and res_md[0]["user"] == "q2"


def test_call_llm_api_openai_compat_mock():
    calls = []

    def mock_transport(url, payload, headers):
        calls.append((url, payload, headers))
        resp_body = json.dumps({"choices": [{"message": {"content": "ok"}}]})
        return 200, resp_body

    prompt = build_prompt(["python basics"], "en", 2)
    res = call_llm_api(
        provider="openai_compat",
        prompt=prompt,
        api_key="test-key",
        model="test-model",
        base_url="https://fake.api/v1",
        transport=mock_transport,
        sleep_fn=lambda s: None,
    )
    assert res == "ok"
    assert len(calls) == 1
    assert calls[0][0] == "https://fake.api/v1/chat_completions" or "chat/completions" in calls[0][0]
    assert calls[0][2]["Authorization"] == "Bearer test-key"


def test_call_llm_api_gemini_mock():
    calls = []

    def mock_transport(url, payload, headers):
        calls.append((url, payload, headers))
        resp_body = json.dumps({"candidates": [{"content": {"parts": [{"text": "gemini ok"}]}}]})
        return 200, resp_body

    res = call_llm_api(
        provider="gemini",
        prompt="hello",
        api_key="gkey",
        model="gemini-2.0-flash",
        transport=mock_transport,
        sleep_fn=lambda s: None,
    )
    assert res == "gemini ok"
    assert "generativelanguage.googleapis.com" in calls[0][0]


def test_call_llm_api_retry_on_429():
    attempts = []

    def flaky_transport(url, payload, headers):
        attempts.append(1)
        if len(attempts) < 2:
            return 429, "rate limit"
        return 200, json.dumps({"choices": [{"message": {"content": "finally ok"}}]})

    sleeps = []
    res = call_llm_api(
        provider="openai_compat",
        prompt="p",
        api_key="k",
        model="m",
        transport=flaky_transport,
        backoff_s=1.0,
        sleep_fn=sleeps.append,
    )
    assert res == "finally ok"
    assert len(attempts) == 2
    assert sleeps == [1.0]


def test_dedup_and_resume_loading(tmp_path):
    out_file = str(tmp_path / "en.jsonl")
    q1 = "What is Python?"
    q1_hash = normalize_question_hash(q1)

    with open(out_file, "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": q1}, {"role": "assistant", "content": "A language"}]}) + "\n")

    seen_hashes, count = load_existing_hashes_and_count(out_file)
    assert count == 1
    assert q1_hash in seen_hashes
