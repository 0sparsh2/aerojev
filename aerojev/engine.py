"""Decision engine: state + typed questions -> calibrated label distributions.

  eng = DecisionEngine("Qwen/Qwen3-0.6B", adapter="runs/qwen3-0.6b/best")
  eng.decide(record_text, [schema.EVENT_TYPE, schema.IMPACT_SEVERITY, ...])

Modes
  tree     : one forward pass per record (prefix + all question branches, 4D mask).
             Dense-attention models only (Qwen2.5, Qwen3).
  unpacked : one sequence per question, batched. Needed for hybrid recurrent
             models (Qwen3.5 Gated DeltaNet), where branches would leak through
             the recurrent state.
"""

import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import prompt, schema
from .prompt import (MAX_RECORD_TOKENS, fit_state, labels_for, materialize, pack, pack_ids, render_branch,
                     render_prefix, render_readout, render_record, render_static, think_block)

HYBRID_MODEL_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_next"}


def load_model(name, device="cpu", dtype=torch.float32, adapter=None):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.to(device).eval()
    return tok, model


def default_mode(model):
    mt = getattr(model.config, "model_type", "")
    text_mt = getattr(getattr(model.config, "text_config", None), "model_type", "")
    return "unpacked" if (mt in HYBRID_MODEL_TYPES or text_mt in HYBRID_MODEL_TYPES) else "tree"


class BaseEngine:
    """Shared decide(): subclasses implement readout_logits(state, qdicts)."""

    temperatures = {}

    def decide(self, state, questions, raw=False):
        """questions: catalogue Question objects or materialized dicts (canonical order/phrasing)."""
        qdicts, keysets = [], []
        for q in questions:
            qd, keys = (q, _keys(q)) if isinstance(q, dict) and "options" in q and "paraphrases" not in q \
                else materialize(q)
            qdicts.append(qd)
            keysets.append(keys)
        logits = self.readout_logits(state, qdicts)
        answers = {}
        for qd, keys, lg in zip(qdicts, keysets, logits):
            T = 1.0 if raw else self.temperatures.get(qd["type"], 1.0)
            probs = torch.softmax(torch.as_tensor(lg, dtype=torch.float32) / T, -1).tolist()
            dist = dict(zip(keys, probs))
            best = max(range(len(probs)), key=lambda i: probs[i])
            a = {"type": qd["type"], "answer": keys[best], "confidence": probs[best], "probabilities": dist}
            if qd["type"] == "score":
                a["expected"] = sum(i * p for i, p in enumerate(probs))
            if qd["type"] == "yesno":
                a["p_yes"] = dist["yes"]
            answers[qd["id"]] = a
        return answers


class DecisionEngine(BaseEngine):
    def __init__(self, name=None, device="cpu", dtype=torch.float32, adapter=None, mode=None,
                 temperatures=None, tok=None, model=None, layout="branch", catalogue=None):
        if model is None:
            tok, model = load_model(name, device, dtype, adapter)
        self.tok, self.model, self.device = tok, model, device
        self.think = think_block(tok)
        self.mode = mode or default_mode(model)
        self.temperatures = temperatures or {}  # per question type, fitted on val
        self._label_cache = {}
        self.layout = layout
        if layout == "catalogue":
            if self.mode != "tree":
                raise ValueError("catalogue layout needs a dense-attention model (tree mode)")
            self.catalogue = [materialize(q)[0] for q in (catalogue or schema.CATALOGUE)]
            static = self.tok.encode(render_static(self.catalogue), add_special_tokens=False)
            self.S = len(static)
            with torch.no_grad():
                self._cache = self.model(input_ids=torch.tensor([static], device=device), use_cache=True,
                                         logits_to_keep=1).past_key_values

    def _label_ids(self, labels):
        key = tuple(labels)
        if key not in self._label_cache:
            self._label_cache[key] = prompt.label_token_ids(self.tok, labels)
        return self._label_cache[key]

    @torch.no_grad()
    def readout_logits(self, state, qdicts):
        """Return list of 1-D tensors: logits over each question's label tokens."""
        state = fit_state(self.tok, state, MAX_RECORD_TOKENS)
        if self.layout == "catalogue":
            return self._catalogue_readout(state, qdicts)
        if self.mode == "tree":
            p = pack(self.tok, state, qdicts, self.think)
            out = self.model(
                input_ids=p["input_ids"][None].to(self.device),
                position_ids=p["position_ids"][None].to(self.device),
                attention_mask=p["mask"][None, None].to(self.device),
                logits_to_keep=torch.tensor(p["readouts"], device=self.device),
            ).logits[0]
            return [out[i, lab].float().cpu() for i, lab in enumerate(p["label_ids"])]
        # unpacked: one unpadded sequence per question (safe for recurrent / M-RoPE models)
        prefix = self.tok.encode(render_prefix(state), add_special_tokens=False)
        res = []
        for q in qdicts:
            ids = prefix + self.tok.encode(render_branch(q, self.think), add_special_tokens=False)
            out = self.model(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1).logits[0, -1]
            res.append(out[self._label_ids(labels_for(q))].float().cpu())
        return res

    def _matches_catalogue(self, qdicts):
        cat = {q["id"]: [o["key"] for o in q["options"]] for q in self.catalogue}
        return all(q["id"] in cat and cat[q["id"]] == [o["key"] for o in q["options"]] for q in qdicts)

    def _catalogue_readout(self, state, qdicts):
        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        labs = [self._label_ids(labels_for(q)) for q in qdicts]
        if not self._matches_catalogue(qdicts):
            # options reordered / unknown question: no cache, catalogue = the asked questions
            p = pack(self.tok, state, qdicts, self.think, layout="catalogue")
            kw = {}
        else:
            p = pack_ids(enc(render_record(state, self.think)), [enc(render_readout(q)) for q in qdicts], labs,
                         offset=self.S)
            kw = {"past_key_values": self._cache, "use_cache": True}
        out = self.model(
            input_ids=p["input_ids"][None].to(self.device),
            position_ids=p["position_ids"][None].to(self.device),
            attention_mask=p["mask"][None, None].to(self.device),
            logits_to_keep=torch.tensor(p["readouts"], device=self.device), **kw,
        ).logits[0]
        if kw:
            extra = self._cache.get_seq_length() - self.S
            if extra > 0:
                self._cache.crop(-extra)  # drop this record's KV, keep the static catalogue
        return [out[i, lab].float().cpu() for i, lab in enumerate(p["label_ids"])]


class GGUFEngine(BaseEngine):
    """llama.cpp CPU backend. The record prefix is evaluated once; each question
    branch is evaluated on top of it and the KV cache is rewound to the prefix."""

    def __init__(self, gguf_path, tokenizer, n_threads=4, n_ctx=4096, n_batch=1024, temperatures=None,
                 layout="branch", catalogue=None):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, n_threads_batch=n_threads,
                         n_batch=n_batch, n_ubatch=n_batch, logits_all=False, verbose=False,
                         # strictly CPU: no layers, KV cache or host ops offloaded to Metal/CUDA
                         n_gpu_layers=0, offload_kqv=False, op_offload=False)
        self.tok = AutoTokenizer.from_pretrained(tokenizer)  # same vocab ids as the GGUF
        self.think = think_block(self.tok)
        self.temperatures = temperatures or {}
        self._label_cache = {}
        self.mode = "gguf"
        self.layout = layout
        self.S = 0
        if layout == "catalogue":
            self.catalogue = [materialize(q)[0] for q in (catalogue or schema.CATALOGUE)]
            self.llm.eval(self.tok.encode(render_static(self.catalogue), add_special_tokens=False))
            self.S = self.llm.n_tokens  # static catalogue stays in the KV cache for the process lifetime

    def _label_ids(self, labels):
        key = tuple(labels)
        if key not in self._label_cache:
            self._label_cache[key] = prompt.label_token_ids(self.tok, labels)
        return self._label_cache[key]

    def _last_logits(self):
        import numpy as np
        # llama-cpp-python >= 0.3 no longer copies logits into .scores unless logits_all=True,
        # so read the last decoded token's row straight from the context
        return np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm.n_vocab(),)).copy()

    def readout_logits(self, state, qdicts):
        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        state = fit_state(self.tok, state, MAX_RECORD_TOKENS)  # keep record + catalogue inside n_ctx
        self.llm.n_tokens = self.S  # eval() drops KV entries at positions >= n_tokens
        if self.layout == "catalogue":
            self.llm.eval(enc(render_record(state, self.think)))
            branches = [enc(render_readout(q)) for q in qdicts]
        else:
            self.llm.eval(enc(render_prefix(state)))
            branches = [enc(render_branch(q, self.think)) for q in qdicts]
        P = self.llm.n_tokens
        res = []
        for q, b in zip(qdicts, branches):
            self.llm.eval(b)
            logits = self._last_logits()
            res.append(torch.tensor(logits[self._label_ids(labels_for(q))], dtype=torch.float32))
            self.llm.n_tokens = P  # rewind to the shared prefix
        return res


def _keys(qd):
    return prompt.YESNO_LABELS if qd["type"] == "yesno" else [o["key"] for o in qd["options"]]
