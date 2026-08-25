"""Teacher client — OpenRouter-compatible /chat/completions, injectable transport.

Owner proxy ya direct OpenRouter dono chalein (base_url configurable).
Offline-testable: transport fn (url, payload, headers) -> (status, body_str)
inject karo. Key/model gating explicit errors deta hai with setup hints.
Retry policy: 429 aur 5xx par exponential backoff; 4xx par turant fail.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ENV_KEY = "RYTH_TEACHER_API_KEY"
ENV_MODEL = "RYTH_TEACHER_MODEL"


class TeacherError(RuntimeError):
    pass


class TeacherConfigError(TeacherError):
    pass


def _default_transport(url: str, payload: dict, headers: dict) -> tuple:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120.0) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:   # error body padhne ke liye swallow
        return e.code, e.read().decode("utf-8", "replace")


class OpenAICompatTeacher:
    """OpenRouter-compatible teacher. api_key/model: arg > env > config-error."""

    def __init__(self, api_key: str | None = None,
                 base_url: str = DEFAULT_BASE_URL, model: str | None = None,
                 model_env: bool = True, transport=None, attempts: int = 4,
                 backoff_s: float = 1.5, sleep=time.sleep):
        self.api_key = api_key or os.environ.get(ENV_KEY)
        self.model = model or (os.environ.get(ENV_MODEL) if model_env else None)
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _default_transport
        self.attempts = attempts
        self.backoff_s = backoff_s
        self._sleep = sleep

    def complete(self, system: str, user: str, *, max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        if not self.api_key:
            raise TeacherConfigError(
                f"teacher API key nahi mili — {ENV_KEY} env set karo ya "
                "OpenAICompatTeacher(api_key=...) pass karo")
        if not self.model:
            raise TeacherConfigError(
                f"teacher model chahiye — {ENV_MODEL} env ya model= kwarg "
                "(owner-proxy ka Nemotron-class backend naam)")
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        last = ""
        for attempt in range(self.attempts):
            status, body = self._transport(url, payload, headers)
            if status == 200:
                try:
                    data = json.loads(body)
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, json.JSONDecodeError,
                        TypeError) as e:
                    raise TeacherError(f"200 par unexpected body ({e}): "
                                       f"{body[:200]}")
            last = f"HTTP {status}: {body[:200]}"
            if status != 429 and status < 500:
                break                                    # 4xx: fail fast
            self._sleep(self.backoff_s * (2 ** attempt))  # 1.5x, 3x, 6x...
        raise TeacherError(f"teacher call fail "
                           f"({attempt + 1}/{self.attempts} tries): {last}")


class FakeTeacher:
    """Tests + --dry-run: substring-routed canned responses, koi network nahi."""

    def __init__(self, responses: dict | None = None,
                 default: str = 'def solved(x):\n    """Done."""\n    return x\n'):
        self.responses = dict(responses or {})
        self.default = default
        self.calls = []

    def complete(self, system: str, user: str, **kw) -> str:
        self.calls.append((system, user))
        for frag, resp in self.responses.items():
            if frag in user:
                return resp
        return self.default
