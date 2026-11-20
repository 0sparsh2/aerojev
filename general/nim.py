"""NVIDIA NIM client for data generation (OpenAI-compatible chat completions), with a response cache.

  from general.nim import NIMClient
  nim = NIMClient("openai/gpt-oss-120b", cache="data/general/nim_cache.jsonl")
  nim.models()                                   # what this key can call (check the generator is live)
  nim.chat([{"role": "user", "content": "..."}], temperature=0.8, seed=1)

* Key: NVIDIA_API_KEY from the environment, a Kaggle Secret (inside kernels) or the git-ignored .env (secret()). It
  is never written to the cache, logs or outputs. Without a key, only cached responses are served; anything else
  raises.
* Cache: every successful response is appended to a JSONL file keyed by a hash of (model, messages, sampling
  settings), so a rerun or a resumed kernel pays only for requests it has not made yet.
* Rate: at most `rpm` requests a minute (the free tier is reported, unofficially, at about 40), with retries and
  backoff on 429 and 5xx.
* Terms (docs/GENERAL_DECISION_PLAN.md, section 6.1): the free trial is for internal evaluation only, so pilot data
  only. Never send confidential data: NVIDIA may use prompts and outputs. Production data needs a paid subscription,
  another provider of the same open model, or self-hosting; only base_url and the key change.
"""

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

BASE_URL = "https://integrate.api.nvidia.com/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def secret(name):
    """An API key: the environment first, then a Kaggle Secret (inside kernels), then the repo's .env file
    (`name = value`, git-ignored). Returns None if none has it. The value is never printed or logged."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            k, _, v = line.partition("=")
            if k.strip().lower() == name.lower() and v.strip():
                return v.strip().strip("\"'")
    return None


class NIMClient:
    def __init__(self, model, cache, base_url=BASE_URL, rpm=35, timeout=120, max_retries=6, key=None):
        """key: None looks it up (secret("NVIDIA_API_KEY")); "" means no key (cache only)."""
        self.model, self.base_url, self.timeout, self.max_retries = model, base_url.rstrip("/"), timeout, max_retries
        self.key = secret("NVIDIA_API_KEY") if key is None else key
        self.interval, self.last, self.lock = 60.0 / rpm, 0.0, threading.Lock()
        self.cache_path, self.cache = cache, {}
        if cache and os.path.exists(cache):
            for line in open(cache):
                r = json.loads(line)
                self.cache[r["key"]] = r["content"]
        self.calls = self.hits = 0

    def _post(self, path, body):
        if not self.key:
            raise RuntimeError("NVIDIA_API_KEY is not set (and the request is not cached)")
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json", "Accept": "application/json"})
        return self._send(req)

    def _send(self, req):
        for attempt in range(self.max_retries):
            with self.lock:  # simple global pacing
                wait = self.last + self.interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                raise RuntimeError(f"NIM HTTP {e.code}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.max_retries - 1:
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                raise

    def models(self):
        if not self.key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        req = urllib.request.Request(self.base_url + "/models", headers={"Authorization": f"Bearer {self.key}"})
        return sorted(m["id"] for m in self._send(req)["data"])

    def chat(self, messages, temperature=0.7, top_p=0.95, max_tokens=1024, seed=None, extra=None):
        """extra: model-specific body fields, e.g. {"chat_template_kwargs": {"enable_thinking": False}} to switch a
        reasoning model's thinking off (otherwise its reasoning can use up max_tokens and leave no answer)."""
        body = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, **({"seed": seed} if seed is not None else {}), **(extra or {})}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        out = self._post("/chat/completions", body)
        choice = out["choices"][0]
        content = choice["message"].get("content")
        self.calls += 1
        if not content:  # never cache an empty answer
            why = choice.get("finish_reason")
            raise RuntimeError(f"empty answer from {self.model} (finish_reason={why}"
                               + (", reasoning used the token budget" if choice["message"].get("reasoning_content")
                                  else "") + ")")
        with self.lock:
            self.cache[key] = content
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "content": content}) + "\n")
        return content


if __name__ == "__main__":  # python -m general.nim --check NAME [NAME ...]: is each secret readable here? (no values)
    import sys
    names = [a for a in sys.argv[1:] if a != "--check"]
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    except Exception as e:
        client = None
        print(f"kaggle_secrets unavailable: {type(e).__name__}: {e}")
    ok = True
    for n in names:
        where = "environment" if os.environ.get(n) else None
        if not where and client:
            try:
                where = "Kaggle Secret" if client.get_secret(n) else None
            except Exception as e:
                print(f"{n}: Kaggle Secret not readable: {type(e).__name__}: {str(e)[:300]}")
        if not where and secret(n):
            where = ".env"
        print(f"{n}: {'found in ' + where if where else 'MISSING'}")
        ok &= bool(where) or n != names[0]  # only the first name is required
    sys.exit(0 if ok else 1)
