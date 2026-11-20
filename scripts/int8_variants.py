"""Which quantization recipe keeps the encoder's accuracy on THIS CPU?

  python scripts/export_encoder.py --model runs/x/model --out onnx/x --keep-fp32
  python scripts/int8_variants.py --dir onnx/x [--out somewhere/writable] [--variants s8,u8,mnb8-a4]

Writes one sub-directory per variant (the quantized graph as model.int8.onnx + decider.json + tokenizer.json, so
eval.evaluate --onnx and scripts/bench_lite.py read it like any export) and prints the x86 CPU features that decide
which int8 kernels ONNX Runtime uses. Without AVX-512 VNNI / AVX-VNNI, u8 x s8 products go through VPMADDUBSW, whose
int16 intermediate can saturate (results/int8diag/int8_x86.md).
"""

import argparse
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_encoder import quantize_int8, quantize_nbits  # noqa: E402

DOWN_PROJ = r"/mlp/(Wo|down_proj)/"  # FFN output projections: ModernBERT / Ettin, UltraBERT

VARIANTS = {
    # dynamic int8, one activation scale per tensor
    "s8": (quantize_int8, {}),                                          # the current recipe
    "s8-rr": (quantize_int8, {"reduce_range": True}),                   # 7-bit weights
    "u8": (quantize_int8, {"weight_type": "quint8"}),                   # u8 x u8
    "s8-constb": (quantize_int8, {"const_b_only": True}),               # attention products stay fp32
    "u8-keepdown": (quantize_int8, {"weight_type": "quint8", "exclude": DOWN_PROJ}),
    # weight-only block quantization (MatMulNBits), int8 embeddings
    "mnb8": (quantize_nbits, {"bits": 8, "accuracy_level": 1}),         # 8-bit weights, fp32 compute
    "mnb8-a4": (quantize_nbits, {"bits": 8, "accuracy_level": 4}),      # 8-bit weights, int8 compute per block
    "mnb4-a4": (quantize_nbits, {"bits": 4, "accuracy_level": 4}),      # 4-bit weights, int8 compute per block
}


def cpu_features():
    if not os.path.exists("/proc/cpuinfo"):
        return platform.processor() or platform.machine()
    info = open("/proc/cpuinfo").read()
    flags = next(l for l in info.splitlines() if l.startswith("flags")).split()
    model = next((l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("model name")), "?")
    want = ["avx2", "avx512f", "avx512bw", "avx512_vnni", "avx_vnni", "amx_int8"]
    return f"{model} | " + " ".join(f"{f}={'yes' if f in flags else 'no'}" for f in want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="an export_encoder output made with --keep-fp32")
    ap.add_argument("--out", help="where to write the variant directories (default: --dir)")
    ap.add_argument("--variants", default="s8,s8-rr,u8,s8-constb")
    args = ap.parse_args()
    print("CPU:", cpu_features(), flush=True)
    fp32 = os.path.join(args.dir, "model.onnx")
    for name in args.variants.split(","):
        out = os.path.join(args.out or args.dir, f"v-{name}")
        os.makedirs(out, exist_ok=True)
        for f in ("decider.json", "tokenizer.json"):
            shutil.copyfile(os.path.join(args.dir, f), os.path.join(out, f))
        fn, kw = VARIANTS[name]
        try:
            fn(fp32, os.path.join(out, "model.int8.onnx"), **kw)
            print(f"{name}: {os.path.getsize(os.path.join(out, 'model.int8.onnx')) / 1e6:.1f} MB", flush=True)
        except Exception as e:  # noqa: BLE001 - report and go on to the next variant
            print(f"{name}: failed: {e!r}", flush=True)
            shutil.rmtree(out)


if __name__ == "__main__":
    main()
