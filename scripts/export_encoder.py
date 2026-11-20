"""Encoder decision model (train/train_encoder.py) -> ONNX files for the torch-free CPU runtime (aerojev.lite).

  python scripts/export_encoder.py --model runs/e150/model --out models/aerojev-ettin-150m \
      --temps results/ft-e150/metrics.json

Writes to --out:
  model.onnx        fp32 graph: encoder + pooling + all 9 heads, logits of every question concatenated
                    (kept only with --keep-fp32: ~4x the int8 file and no more accurate)
  model.int8.onnx   dynamic int8: the encoder's weight matrices and the token embedding table stored as int8
                    (per output channel); activations are quantised on the fly. Heads and pooling stay fp32.
  model.int4.onnx   (--int4) the x86 file: 4-bit block-wise weights for the encoder's matrices (int8 compute per
                    block of 32), int8 embeddings. Lossless on x86, where model.int8.onnx is not.
  decider.json      question ids, option keys, output offsets, max length, temperatures
  tokenizer.json    the fast tokenizer (read by the `tokenizers` library; no transformers needed)

The fp32 graph is checked against PyTorch on real records before anything is quantised.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train_encoder import KEYS, EncoderDecider  # noqa: E402


class Flat(nn.Module):
    """dict of per-question logits -> one [batch, sum of option counts] tensor, in KEYS order."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids, attention_mask)
        return torch.cat([out[q] for q in KEYS], -1)


def decision_nodes(fp32):
    """Heads and pooling scores: tiny, and they sit right at the decision, so they stay fp32."""
    import onnx
    return [n.name for n in onnx.load(fp32).graph.node if n.name.startswith(("/m/heads", "/m/attn/"))]


def quantize_int8(fp32, out, weight_type="qint8", reduce_range=False, const_b_only=False, exclude=None):
    """Dynamic int8: the encoder's weight matrices and the token embedding table as int8 (per output channel),
    activations quantised to uint8 on the fly (one scale per tensor).
    weight_type="quint8" (u8 x u8) or reduce_range=True (7-bit weights) avoid the u8 x s8 int16 saturation of ONNX
    Runtime's VPMADDUBSW kernels on x86 CPUs that have AVX2 / AVX-512 but no VNNI; const_b_only=True leaves the
    activation x activation products (attention scores, attention-weighted values) in fp32; exclude: a regex of node
    names that stay fp32 as well (e.g. the FFN down-projections, whose inputs carry the outliers)."""
    import re
    from onnxruntime.quantization import QuantType, quantize_dynamic
    keep = decision_nodes(fp32)
    if exclude:
        import onnx
        keep += [n.name for n in onnx.load(fp32).graph.node if n.op_type == "MatMul" and re.search(exclude, n.name)]
        print(f"  {len(keep)} nodes kept in fp32 (pattern {exclude!r})", flush=True)
    quantize_dynamic(fp32, out, weight_type={"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}[weight_type],
                     per_channel=True, reduce_range=reduce_range, op_types_to_quantize=["MatMul", "Gather"],
                     nodes_to_exclude=keep, extra_options={"MatMulConstBOnly": const_b_only})


def quantize_nbits(fp32, out, bits=8, block_size=32, accuracy_level=4):
    """Weight-only block quantization (MatMulNBits) of the encoder's weight matrices, then int8 embeddings.
    Activations are not quantized per tensor: accuracy_level=1 computes in fp32; accuracy_level=4 quantizes the
    activations to int8 per block of `block_size` values along each row (per token), which keeps outliers local."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(fp32), bits=bits, block_size=block_size, is_symmetric=True,
                             accuracy_level=accuracy_level, nodes_to_exclude=decision_nodes(fp32))
    q.process()
    tmp = out + ".mm.onnx"
    q.model.save_model_to_file(tmp, use_external_data_format=False)
    quantize_dynamic(tmp, out, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Gather"])
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a train_encoder model/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", help="metrics.json (its 'temperatures') or a plain {type: T} json")
    ap.add_argument("--int4", action="store_true", help="also write 4-bit block-wise weights (model.int4.onnx)")
    ap.add_argument("--keep-fp32", action="store_true", help="keep model.onnx (the int8 file is built from it)")
    ap.add_argument("--data", default="data/processed_v2")
    ap.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    ap.add_argument("--reduce-range", action="store_true")
    ap.add_argument("--const-b-only", action="store_true", help="quantize only MatMuls with a weight matrix")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = EncoderDecider.load(args.model, attn_implementation="eager").eval()
    flat = Flat(model).eval()

    rows = [json.loads(l) for l in open(os.path.join(args.data, "gold.jsonl"))][:8]
    enc = tok([r["state"] for r in rows], padding=True, return_tensors="pt")
    fp32 = os.path.join(args.out, "model.onnx")
    with torch.no_grad():
        ref = flat(enc["input_ids"], enc["attention_mask"]).numpy()
        torch.onnx.export(flat, (enc["input_ids"], enc["attention_mask"]), fp32, dynamo=False, opset_version=17,
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                                        "attention_mask": {0: "batch", 1: "tokens"}, "logits": {0: "batch"}})

    import onnxruntime as ort
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    feed = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    err = float(np.abs(sess.run(None, feed)[0] - ref).max())
    # also one record at a time, unpadded, at other lengths than the traced batch: that is how aerojev.lite runs it,
    # and a shape fixed at trace time would only show up here
    more = [json.loads(l)["state"] for l in open(os.path.join(args.data, "test.jsonl"))][:6]
    for text in [r["state"] for r in rows[:2]] + more:
        e = tok(text, return_tensors="pt")
        with torch.no_grad():
            want = flat(e["input_ids"], e["attention_mask"]).numpy()
        got = sess.run(None, {"input_ids": e["input_ids"].numpy(), "attention_mask": e["attention_mask"].numpy()})[0]
        err = max(err, float(np.abs(got - want).max()))
    print(f"fp32 ONNX vs PyTorch: max |logit diff| {err:.2e} (padded batch of {len(rows)} + {len(more) + 2} single records)")
    assert err < 1e-3, "the ONNX graph does not match PyTorch"
    del sess

    quantize_int8(fp32, os.path.join(args.out, "model.int8.onnx"), args.weight_type, args.reduce_range, args.const_b_only)
    if args.int4:
        # the x86 file: weight-only 4-bit with per-block int8 compute (accuracy_level 4) was lossless on x86 CPUs
        # without VNNI, where the dynamic int8 file breaks (results/int8diag/int8_x86.md)
        quantize_nbits(fp32, os.path.join(args.out, "model.int4.onnx"), bits=4, block_size=32, accuracy_level=4)

    meta = json.load(open(os.path.join(args.model, "decider.json")))
    sizes = [len(KEYS[q]) for q in KEYS]
    meta["offsets"] = {q: [int(sum(sizes[:i])), int(sum(sizes[:i + 1]))] for i, q in enumerate(KEYS)}
    if args.temps:
        t = json.load(open(args.temps))
        meta["temperatures"] = t.get("temperatures", t)
    json.dump(meta, open(os.path.join(args.out, "decider.json"), "w"), indent=1)
    shutil.copyfile(os.path.join(args.model, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    if not args.keep_fp32:
        os.remove(fp32)
    for f in sorted(os.listdir(args.out)):
        print(f"  {f:20s} {os.path.getsize(os.path.join(args.out, f)) / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
