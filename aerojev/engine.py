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
                     render_prefix, render_readout, render_record, render_static, render_static_nocat,
                     think_block)

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

    def _matches_catalogue(self, qdicts):
        cat = {q["id"]: [o["key"] for o in q["options"]] for q in self.catalogue}
        return all(q["id"] in cat and cat[q["id"]] == [o["key"] for o in q["options"]] for q in qdicts)

    def _check_nocat(self, qdicts):
        """A nocat model never sees question text or options: it only knows its trained questions, each with its
        options in the trained (canonical) order. Anything else would get a meaningless answer, so refuse it."""
        if self.layout == "nocat" and not self._matches_catalogue(qdicts):
            known = ", ".join(q["id"] for q in self.catalogue)
            raise ValueError("this model was trained without a question list in the prompt (nocat layout); it "
                             f"answers only its trained questions, with options in schema order: {known}")

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
        if layout in ("catalogue", "nocat"):
            if self.mode != "tree":
                raise ValueError(f"{layout} layout needs a dense-attention model (tree mode)")
            self.catalogue = [materialize(q)[0] for q in (catalogue or _default_catalogue(layout))]
            static = self.tok.encode(_static(layout, self.catalogue), add_special_tokens=False)
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
        if self.layout in ("catalogue", "nocat"):
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

    def _catalogue_readout(self, state, qdicts):
        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        labs = [self._label_ids(labels_for(q)) for q in qdicts]
        self._check_nocat(qdicts)
        if not self._matches_catalogue(qdicts):
            # options reordered / unknown question: no cache, catalogue = the asked questions
            p = pack(self.tok, state, qdicts, self.think, layout=self.layout)
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
    """llama.cpp CPU backend.

    catalogue / nocat layouts (batched=True, the default): one llama_decode call per record answers every question.
    llama.cpp's multi-sequence batches stand in for the tree mask: the static prefix belongs to every sequence, the
    record's tokens join every asked question's sequence, and each readout joins only its own, so a readout sees the
    prefix, the record and itself. Each llama_decode call reads all the weights, which dominates on a CPU once the
    catalogue is cached, so this is much faster than one call per question.
    branch layout (or batched=False): the record prefix is evaluated once; each question branch is evaluated on top of
    it and the KV cache is rewound to the prefix."""

    def __init__(self, gguf_path, tokenizer, n_threads=4, n_ctx=4096, n_batch=1024, temperatures=None,
                 layout="branch", catalogue=None, batched=True):
        from llama_cpp import Llama
        batched = batched and layout in ("catalogue", "nocat")
        small = dict(n_ctx=256, n_batch=256, n_ubatch=256) if batched else {}  # the batched path makes its own context
        self.llm = Llama(model_path=gguf_path, **{"n_ctx": n_ctx, "n_batch": n_batch, "n_ubatch": n_batch, **small},
                         n_threads=n_threads, n_threads_batch=n_threads, logits_all=False, verbose=False,
                         # strictly CPU: no layers, KV cache or host ops offloaded to Metal/CUDA
                         n_gpu_layers=0, offload_kqv=False, op_offload=False)
        # free llama.cpp objects before interpreter teardown; llama-cpp-python 0.3.16 otherwise raises at exit
        __import__("atexit").register(self.llm.close)
        self.tok = AutoTokenizer.from_pretrained(tokenizer)  # same vocab ids as the GGUF
        self.think = think_block(self.tok)
        self.temperatures = temperatures or {}
        self._label_cache = {}
        self.mode = "gguf"
        self.layout = layout
        self.S = 0
        self.batched = False
        if layout in ("catalogue", "nocat"):
            self.catalogue = [materialize(q)[0] for q in (catalogue or _default_catalogue(layout))]
            static = self.tok.encode(_static(layout, self.catalogue), add_special_tokens=False)
            if batched:
                self._init_batched(static, n_ctx, n_batch)
            else:
                self.llm.eval(static)
                self.S = self.llm.n_tokens  # static catalogue stays in the KV cache for the process lifetime

    def _init_batched(self, static, n_ctx, n_batch):
        import contextlib

        import llama_cpp
        from llama_cpp import _internals
        self._nseq = len(self.catalogue)  # one sequence per question that can be asked in one call
        params = llama_cpp.llama_context_params.from_buffer_copy(self.llm.context_params)
        params.n_ctx, params.n_batch, params.n_ubatch = n_ctx, n_batch, n_batch
        params.n_seq_max, params.kv_unified = self._nseq, True  # unified: a token may belong to several sequences
        # entered on the Llama's own exit stack, so they are freed before the model they belong to
        ctx = self.llm._stack.enter_context(contextlib.closing(
            _internals.LlamaContext(model=self.llm._model, params=params, verbose=False)))
        self._batch = llama_cpp.llama_batch_init(n_batch, 0, self._nseq)
        self.llm._stack.callback(llama_cpp.llama_batch_free, self._batch)
        self._ctx, self._mem, self._n_batch = ctx.ctx, ctx.memory, n_batch
        self._decode([(t, i, range(self._nseq), False) for i, t in enumerate(static)])
        self.S = len(static)  # the static part stays in the KV cache, in every sequence, for the process lifetime
        self.batched = True

    def _decode(self, items):
        """items: (token, position, sequence ids, want logits), decoded in chunks of n_batch tokens."""
        import llama_cpp
        b = self._batch
        for c in range(0, len(items), self._n_batch):
            chunk = items[c:c + self._n_batch]
            b.n_tokens = len(chunk)
            for j, (t, pos, seqs, want) in enumerate(chunk):
                b.token[j], b.pos[j], b.logits[j] = t, pos, want
                b.n_seq_id[j] = len(seqs)
                for k, sq in enumerate(seqs):
                    b.seq_id[j][k] = sq
            if llama_cpp.llama_decode(self._ctx, b) != 0:
                raise RuntimeError("llama_decode failed (KV cache full? record too long for n_ctx)")

    def _batched_readout(self, state, qdicts):
        import llama_cpp
        import numpy as np
        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        rec = enc(render_record(state, self.think))
        res = []
        for c in range(0, len(qdicts), self._nseq):  # at most one question per sequence in a call
            qs = qdicts[c:c + self._nseq]
            seqs = list(range(len(qs)))
            rec_items = [(t, self.S + j, seqs, False) for j, t in enumerate(rec)]
            ro_items, ends = [], []
            for sq, q in zip(seqs, qs):
                b = enc(render_readout(q))
                ro_items += [(t, self.S + len(rec) + j, (sq,), j == len(b) - 1) for j, t in enumerate(b)]
                ends.append(len(ro_items) - 1)
            if len(rec_items) + len(ro_items) > self._n_batch:  # long record: decode it first, readouts last
                self._decode(rec_items)
                rec_items = []
            self._decode(rec_items + ro_items)  # usually the whole record and every readout in one call
            n_vocab = self.llm.n_vocab()
            for q, e in zip(qs, ends):
                row = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(self._ctx, len(rec_items) + e), shape=(n_vocab,))
                res.append(torch.tensor(row[self._label_ids(labels_for(q))], dtype=torch.float32))
            for sq in seqs:
                llama_cpp.llama_memory_seq_rm(self._mem, sq, self.S, -1)  # forget the record, keep the static part
        return res

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
        self._check_nocat(qdicts)
        state = fit_state(self.tok, state, MAX_RECORD_TOKENS)  # keep record + catalogue inside n_ctx
        if self.batched:
            return self._batched_readout(state, qdicts)
        self.llm.n_tokens = self.S  # eval() drops KV entries at positions >= n_tokens
        if self.layout in ("catalogue", "nocat"):
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


def _default_catalogue(layout):
    """catalogue layout: every question, held-out ones included; nocat: only the trained questions exist."""
    return schema.TRAIN_QUESTIONS if layout == "nocat" else schema.CATALOGUE


def _static(layout, catalogue):
    """Cached system part: the full question catalogue, or (nocat) only the short system line."""
    return render_static_nocat() if layout == "nocat" else render_static(catalogue)


def _keys(qd):
    return prompt.YESNO_LABELS if qd["type"] == "yesno" else [o["key"] for o in qd["options"]]
