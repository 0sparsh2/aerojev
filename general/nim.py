"""NVIDIA NIM client for data generation (OpenAI-compatible chat completions), with a response cache.

  from general.nim import NIMClient
  nim = NIMClient("openai/gpt-oss-120b", cache="data/general/nim_cache.jsonl")
  nim.models()                                   # what this key can call (check the generator is live)
  nim.chat([{"role": "user", "content": "..."}], temperature=0.8, seed=1)

* Key: the NVIDIA_API_KEY environment variable (a Kaggle Secret inside kernels). It is never written to the cache,
  logs or outputs. Without a key, only cached responses are served; anything else raises.
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


class NIMClient:
    def __init__(self, model, cache, base_url=BASE_URL, rpm=35, timeout=120, max_retries=6):
        self.model, self.base_url, self.timeout, self.max_retries = model, base_url.rstrip("/"), timeout, max_retries
        self.key = os.environ.get("NVIDIA_API_KEY")
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

    def chat(self, messages, temperature=0.7, top_p=0.95, max_tokens=1024, seed=None):
        body = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, **({"seed": seed} if seed is not None else {})}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        out = self._post("/chat/completions", body)
        content = out["choices"][0]["message"]["content"]
        self.calls += 1
        with self.lock:
            self.cache[key] = content
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "content": content}) + "\n")
        return content
