"""Merge a LoRA adapter into its base model and export GGUF files for llama.cpp (CPU inference).

  python -m aerojev.export --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen2.5-0.5b/best \\
      --out models --name aerojev-qwen2.5-0.5b

The first run clones llama.cpp and builds its `llama-quantize` tool (CPU only), which needs git, cmake
and a C++ compiler. Q8_0 is written directly by llama.cpp's converter; Q4_K_M and other k-quants are
made from an intermediate f16 file with `llama-quantize`.
"""

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_LLAMA_DIR = os.path.expanduser("~/.cache/aerojev/llama.cpp")


def ensure_llama_cpp(llama_dir=DEFAULT_LLAMA_DIR):
    """Return the path to `llama-quantize`, cloning and building llama.cpp the first time."""
    qbin = os.path.join(llama_dir, "build", "bin", "llama-quantize")
    if os.path.exists(qbin):
        return qbin
    if not os.path.exists(llama_dir):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", llama_dir], check=True)
    # the converter script imports llama.cpp's own gguf package
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", os.path.join(llama_dir, "gguf-py")], check=True)
    build = os.path.join(llama_dir, "build")
    subprocess.run(["cmake", "-S", llama_dir, "-B", build, "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DLLAMA_CURL=OFF"],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", build, "--target", "llama-quantize", "-j", "4"], check=True,
                   stdout=subprocess.DEVNULL)
    return qbin


def export_gguf(model, tokenizer, out_dir, name, llama_dir=DEFAULT_LLAMA_DIR, quants=("q8_0", "q4_k_m")):
    """Write <out_dir>/<name>-<quant>.gguf for a merged (no adapter) transformers model; returns {quant: path}."""
    os.makedirs(out_dir, exist_ok=True)
    qbin = ensure_llama_cpp(llama_dir)
    convert = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    hf_dir = os.path.join(out_dir, f".{name}-hf")  # temporary safetensors copy the converter reads
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    quiet = {"check": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    f16 = os.path.join(out_dir, f"{name}-f16.gguf")
    paths = {}
    try:
        subprocess.run([sys.executable, convert, hf_dir, "--outtype", "f16", "--outfile", f16], **quiet)
        for q in quants:
            path = os.path.join(out_dir, f"{name}-{q}.gguf")
            if q == "q8_0":
                subprocess.run([sys.executable, convert, hf_dir, "--outtype", "q8_0", "--outfile", path], **quiet)
            else:
                subprocess.run([qbin, f16, path, q.upper()], **quiet)
            paths[q] = path
    finally:
        shutil.rmtree(hf_dir, ignore_errors=True)
        if os.path.exists(f16):
            os.remove(f16)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id or path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (e.g. runs/<run>/best)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--name", required=True, help="file name prefix, e.g. aerojev-qwen3-0.6b")
    ap.add_argument("--llama-dir", default=DEFAULT_LLAMA_DIR)
    ap.add_argument("--quants", default="q8_0,q4_k_m")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    paths = export_gguf(model, AutoTokenizer.from_pretrained(args.model), args.out, args.name, args.llama_dir,
                        tuple(args.quants.split(",")))
    for q, p in paths.items():
        print(f"{q:7s} {os.path.getsize(p) / 1e6:6.0f} MB  {p}")


if __name__ == "__main__":
    main()
