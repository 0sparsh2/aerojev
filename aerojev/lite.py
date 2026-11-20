"""AeroJev lite: the encoder decision model on ONNX Runtime, for small CPUs.

No PyTorch and no transformers: numpy, onnxruntime and tokenizers only, so the process stays small. The model
directory comes from scripts/export_encoder.py (model.int8.onnx / model.int4.onnx / model.onnx, decider.json,
tokenizer.json).

  from aerojev.lite import LiteEngine
  eng = LiteEngine("models/aerojev-ettin-150m")      # weights file chosen by CPU: int8 on ARM, int4 / fp32 on x86
  eng.decide(record_text)                             # every trained question -> answer, confidence, probabilities

One forward pass over the record answers all 9 trained questions. Like the catalogue-free Qwen model it knows only
its trained questions, each with its options in schema order; anything else is refused.
"""

import json
import os
import platform
import warnings

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from . import schema

YESNO = ["yes", "no"]


def _qdict(q):
    """schema Question -> the plain dict the engines take (canonical text and option order)."""
    return {"id": q.id, "type": q.type, "text": q.text,
            "options": [{"key": o.key, "description": o.description} for o in q.options]}


def _keys(qd):
    return YESNO if qd["type"] == "yesno" else [o["key"] for o in qd["options"]]


def is_x86():
    return platform.machine().lower() in ("x86_64", "amd64")


def x86_without_vnni():
    """True on a Linux x86 CPU with neither AVX-512 VNNI nor AVX-VNNI. There, ONNX Runtime's u8 x s8 int8 kernels
    (VPMADDUBSW) saturate and the dynamic-int8 file loses 30-55 accuracy points (measured on Kaggle's Skylake and
    AVX2 Xeons, results/int8diag/int8_x86.md). Apple Silicon / ARM are not affected. Other OSes: not checked."""
    if not is_x86():
        return False
    try:
        flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
    except (OSError, StopIteration):
        return False
    return "avx512_vnni" not in flags and "avx_vnni" not in flags


class LiteEngine:
    layout = mode = "encoder"

    def __init__(self, path, temperatures=None, threads=None, weights=None):
        """weights: "int8", "int4" or "fp32". Default: on ARM (Apple Silicon) int8, then int4, then fp32; on x86 int4,
        then fp32, then int8. The dynamic int8 file is verified accurate on ARM only: on x86 CPUs without VNNI it loses
        30-55 points, and the saturation-free u8 variant still lost 2-5 (Ettin) / 15-20 (UltraBERT); x86 with VNNI was
        not tested. The int4 file (weight-only, int8 compute per block) and fp32 are lossless on x86
        (results/int8diag/int8_x86.md)."""
        meta = json.load(open(os.path.join(path, "decider.json")))
        self.keys, self.types, self.offsets = meta["keys"], meta["types"], meta["offsets"]
        self.temperatures = temperatures if temperatures is not None else meta.get("temperatures", {})
        self.catalogue = [_qdict(q) for q in schema.TRAIN_QUESTIONS]
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.tok.enable_truncation(meta["max_len"])
        self.tok.no_padding()
        files = {"int8": "model.int8.onnx", "int4": "model.int4.onnx", "fp32": "model.onnx"}
        if weights is None:
            order = ("int4", "fp32", "int8") if is_x86() else ("int8", "int4", "fp32")
            weights = next(w for w in order if os.path.exists(os.path.join(path, files[w])))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False  # M4, int8, 1 thread: peak 443 -> 355 MB for ~4 ms more per record
        self.weights = weights
        if weights == "int8" and x86_without_vnni():
            warnings.warn("This x86 CPU has no VNNI: the int8 file's answers are badly wrong here (30-55 accuracy "
                          "points lost, results/int8diag/int8_x86.md). Use the x86 file model.int4.onnx "
                          "(scripts/export_encoder.py --int4) or fp32 (--keep-fp32).", RuntimeWarning, stacklevel=2)
        self.sess = ort.InferenceSession(os.path.join(path, files[weights]), so, providers=["CPUExecutionProvider"])

    def _check(self, qdicts):
        cat = {q["id"]: [o["key"] for o in q["options"]] for q in self.catalogue}
        if not all(q["id"] in cat and cat[q["id"]] == [o["key"] for o in q["options"]] for q in qdicts):
            raise ValueError("this encoder model answers only its trained questions, with options in schema order: "
                             + ", ".join(cat))

    _check_nocat = _check  # the server's name for this check

    def all_logits(self, state):
        """One record -> {question id: logits over its option keys (schema order)}."""
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        ids = np.asarray([self.tok.encode(text).ids], dtype=np.int64)
        out = self.sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        return {q: out[a:b] for q, (a, b) in self.offsets.items()}

    def readout_logits(self, state, qdicts):
        self._check(qdicts)
        lg = self.all_logits(state)
        return [lg[q["id"]] for q in qdicts]

    def decide(self, state, questions=None, raw=False):
        """questions: schema Question objects or canonical dicts; default every trained question.
        Same output as aerojev.engine.BaseEngine.decide."""
        qdicts = [q if isinstance(q, dict) else _qdict(q) for q in (questions or schema.TRAIN_QUESTIONS)]
        answers = {}
        for qd, lg in zip(qdicts, self.readout_logits(state, qdicts)):
            keys = _keys(qd)
            T = 1.0 if raw else self.temperatures.get(qd["type"], 1.0)
            z = np.asarray(lg, dtype=np.float64) / T
            p = np.exp(z - z.max())
            p /= p.sum()
            best = int(p.argmax())
            a = {"type": qd["type"], "answer": keys[best], "confidence": float(p[best]),
                 "probabilities": dict(zip(keys, p.tolist()))}
            if qd["type"] == "score":
                a["expected"] = float(sum(i * x for i, x in enumerate(p)))
            if qd["type"] == "yesno":
                a["p_yes"] = a["probabilities"]["yes"]
            answers[qd["id"]] = a
        return answers
