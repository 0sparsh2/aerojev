"""AeroJev HTTP API and test page (standard library only), CPU inference.

  python server.py --preset best      # Qwen3-0.6B LoRA (data v2), GGUF Q8_0, catalogue layout
  python server.py --preset fast      # Qwen2.5-0.5B LoRA, GGUF Q8_0, catalogue layout
  open http://127.0.0.1:8080          # the test page (ui/index.html)

  python server.py --gguf models/aerojev-qwen3-0.6b-cat-v2-q8_0.gguf --tokenizer Qwen/Qwen3-0.6B --layout catalogue
  python server.py --model Qwen/Qwen3-0.6B --adapter runs/qwen3-0.6b/best

  curl -s localhost:8080/v1/decide -d '{"state": "Kestrel says the 718 disks slip 6 weeks; 2 weeks on hand; sole source."}'
  curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": ["impact_severity", "recommended_action"]}'
  curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": [{"id": "needs_buyer_call", "type": "yesno",
        "text": "Should a buyer call the supplier today?"}]}'

Questions are catalogue ids or ad-hoc question objects ({id, type, text, options:[{key, description}]}).
Ad-hoc questions work with either layout; in the catalogue layout they bypass the static KV cache.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from aerojev import schema
from aerojev.prompt import materialize

ENGINE = None
MODEL_NAME = ""
LOCK = Lock()  # one forward pass at a time: the KV cache is shared state
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")

PRESETS = {  # weights come from training + `python -m aerojev.export` (they are not in git)
    "best": dict(gguf="models/aerojev-qwen3-0.6b-cat-v2-q8_0.gguf", tokenizer="Qwen/Qwen3-0.6B",
                 temps="models/temps-qwen3-0.6b-cat-v2.json", label="Qwen3-0.6B + LoRA (data v2), Q8_0"),
    "fast": dict(gguf="models/aerojev-qwen2.5-0.5b-cat-q8_0.gguf", tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
                 temps="models/temps-qwen2.5-0.5b-cat.json", label="Qwen2.5-0.5B + LoRA, Q8_0"),
}


def to_question(q):
    if isinstance(q, str):
        if q not in schema.BY_ID:
            raise ValueError(f"unknown question id {q!r}")
        return materialize(schema.BY_ID[q])[0]
    if q.get("type") not in ("choice", "score", "yesno"):
        raise ValueError("question.type must be choice | score | yesno")
    opts = q.get("options", [])
    if q["type"] != "yesno" and not 2 <= len(opts) <= 26:
        raise ValueError("choice/score questions need 2-26 options")
    return {"id": q["id"], "type": q["type"], "text": q["text"],
            "options": [{"key": o["key"], "description": o["description"]} for o in opts]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj, content_type="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(UI, "rb").read(), "text/html; charset=utf-8")
        if self.path == "/health":
            return self._send(200, {"ok": True, "model": MODEL_NAME, "layout": ENGINE.layout, "mode": ENGINE.mode,
                                    "engine": "llama.cpp" if ENGINE.mode == "gguf" else "torch",
                                    "calibrated": bool(ENGINE.temperatures)})
        if self.path == "/v1/questions":
            return self._send(200, [materialize(q)[0] for q in schema.CATALOGUE])
        if self.path == "/v1/examples":
            from ui.examples import examples
            return self._send(200, examples())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/decide":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            state = req["state"]
            if not state:
                raise ValueError("state is empty")
            qs = [to_question(q) for q in req.get("questions", [q.id for q in schema.CATALOGUE])]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            return self._send(422, {"error": str(e)})
        t0 = time.time()
        with LOCK:
            answers = ENGINE.decide(state, qs)
        self._send(200, {"answers": answers, "latency_ms": round(1000 * (time.time() - t0), 1)})

    def log_message(self, fmt, *args):
        pass


def main():
    global ENGINE, MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), help="a trained model from models/ with its settings")
    ap.add_argument("--model", help="HF model id (tokenizer for --gguf)")
    ap.add_argument("--tokenizer")
    ap.add_argument("--adapter")
    ap.add_argument("--gguf")
    ap.add_argument("--layout", choices=["branch", "catalogue"], default="branch")
    ap.add_argument("--temps", help="fitted temperatures: an eval metrics.json or a {type: T} file")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    if args.preset:
        p = PRESETS[args.preset]
        args.gguf, args.tokenizer, args.temps, args.layout = p["gguf"], p["tokenizer"], p["temps"], "catalogue"
        if not os.path.exists(args.gguf):
            raise SystemExit(f"{args.gguf} not found: train and export a model first (see README, steps 3-5)")
    import torch
    torch.set_num_threads(args.threads)
    if args.gguf:
        from aerojev.engine import GGUFEngine
        ENGINE = GGUFEngine(args.gguf, args.tokenizer or args.model, n_threads=args.threads, layout=args.layout)
    else:
        from aerojev.engine import DecisionEngine
        ENGINE = DecisionEngine(args.model, adapter=args.adapter, layout=args.layout)
    if args.temps:
        t = json.load(open(args.temps))
        ENGINE.temperatures = t.get("temperatures", t)
    MODEL_NAME = PRESETS[args.preset]["label"] if args.preset else os.path.basename(args.gguf or args.adapter or args.model)
    print(f"AeroJev ({MODEL_NAME}, {args.layout} layout) on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
