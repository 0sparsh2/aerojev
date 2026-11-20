"""Kaggle kernel entry point: run one job queue per GPU, in parallel.

The kernel script is this file with JOBS filled in by kaggle/make_kernel.py.
Code + data arrive as /kaggle/input/aerojev-bundle/aerojev_bundle.tar.gz.
Model weights are pulled from the HF Hub inside Kaggle, so nothing large is
downloaded to the local machine; only results, adapters and GGUF files are
written to /kaggle/working/out.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

JOBS = {'0': [{'kind': 'cmd', 'cmd': '{PY} scripts/prune.py --drop top:8 --out /tmp/aerojev/pruned/d16top && mkdir -p {OUT}/pruned/d16top && cp /tmp/aerojev/pruned/d16top/prune_info.json {OUT}/pruned/d16top/'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/d16top', 'out': 'runs/p-d16top', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2']}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/d16top', 'out': 'results/ft-p-d16top', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection', 'adapter': 'runs/p-d16top/best'}, {'kind': 'cmd', 'cmd': '{PY} scripts/prune.py --drop top:8 --ffn 2560 --out /tmp/aerojev/pruned/d16w && mkdir -p {OUT}/pruned/d16w && cp /tmp/aerojev/pruned/d16w/prune_info.json {OUT}/pruned/d16w/'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/d16w', 'out': 'runs/p-d16w', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2']}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/d16w', 'out': 'results/ft-p-d16w', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection', 'adapter': 'runs/p-d16w/best'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/d16w', 'out': 'runs/p-d16w-full-2e-5', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2', '--lr', '2e-5', '--full-ft']}, {'kind': 'eval', 'model': '/kaggle/working/out/runs/p-d16w-full-2e-5/best', 'out': 'results/ft-p-d16w-full-2e-5', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection'}, {'kind': 'cmd', 'cmd': '{PY} -m aerojev.export --model {OUT}/runs/p-d16w-full-2e-5/best --out {OUT}/gguf --name aerojev-p-d16w-full-2e-5 --quants q8_0 && rm -f {OUT}/runs/p-d16w-full-2e-5/best/model.safetensors'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/d16w', 'out': 'runs/p-d16w-full-6e-5', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2', '--lr', '6e-5', '--full-ft']}, {'kind': 'eval', 'model': '/kaggle/working/out/runs/p-d16w-full-6e-5/best', 'out': 'results/ft-p-d16w-full-6e-5', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection'}, {'kind': 'cmd', 'cmd': '{PY} -m aerojev.export --model {OUT}/runs/p-d16w-full-6e-5/best --out {OUT}/gguf --name aerojev-p-d16w-full-6e-5 --quants q8_0 && rm -f {OUT}/runs/p-d16w-full-6e-5/best/model.safetensors'}], '1': [{'kind': 'train', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'out': 'runs/l4-nocat-12k-seed1', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2', '--seed', '1']}, {'kind': 'eval', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'out': 'results/ft-l4-nocat-12k-seed1', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection', 'adapter': 'runs/l4-nocat-12k-seed1/best'}, {'kind': 'cmd', 'cmd': '{PY} scripts/prune.py --drop auto:8 --out /tmp/aerojev/pruned/d16auto && mkdir -p {OUT}/pruned/d16auto && cp /tmp/aerojev/pruned/d16auto/prune_info.json {OUT}/pruned/d16auto/'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/d16auto', 'out': 'runs/p-d16auto', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2']}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/d16auto', 'out': 'results/ft-p-d16auto', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection', 'adapter': 'runs/p-d16auto/best'}, {'kind': 'cmd', 'cmd': '{PY} scripts/prune.py --ffn 2560 --out /tmp/aerojev/pruned/w2560 && mkdir -p {OUT}/pruned/w2560 && cp /tmp/aerojev/pruned/w2560/prune_info.json {OUT}/pruned/w2560/'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/w2560', 'out': 'runs/p-w2560', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '1.2']}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/w2560', 'out': 'results/ft-p-w2560', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection', 'adapter': 'runs/p-w2560/best'}, {'kind': 'cmd', 'cmd': '{PY} scripts/prune.py --adapter {GGUF:runs/l4-nocat-12k/best/adapter_model.safetensors} --drop top:8 --out /tmp/aerojev/pruned/t-d16 && mkdir -p {OUT}/pruned/t-d16 && cp /tmp/aerojev/pruned/t-d16/prune_info.json {OUT}/pruned/t-d16/'}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/t-d16', 'out': 'results/t-d16-noheal', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection'}, {'kind': 'cmd', 'cmd': '{PY} scripts/prune.py --adapter {GGUF:runs/l4-nocat-12k/best/adapter_model.safetensors} --drop top:8 --ffn 2560 --out /tmp/aerojev/pruned/t-d16w && mkdir -p {OUT}/pruned/t-d16w && cp /tmp/aerojev/pruned/t-d16w/prune_info.json {OUT}/pruned/t-d16w/'}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/t-d16w', 'out': 'results/t-d16w-noheal', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection'}, {'kind': 'train', 'model': '/tmp/aerojev/pruned/t-d16w', 'out': 'runs/t-d16w-heal3k', 'args': ['--layout', 'nocat', '--shared-catalogue', '--min-q', '9', '--lora-dropout', '0', '--cached-val', '--data', 'data/processed_v2', '--epochs', '0.3']}, {'kind': 'eval', 'model': '/tmp/aerojev/pruned/t-d16w', 'out': 'results/ft-t-d16w-heal3k', 'layout': 'nocat', 'data': 'data/processed_v2', 'sets': 'val,test,test_v2,gold,gold2,injection', 'adapter': 'runs/t-d16w-heal3k/best'}]}  # generated: {"0": [job, ...], "1": [job, ...]}

WORK = "/kaggle/working"
PROJ = "/tmp/aerojev/proj"
LLAMA = "/tmp/aerojev/llama.cpp"
LLAMA_LOCK = threading.Lock()
OUT = os.path.join(WORK, "out")


def sh(cmd, log, env=None, cwd=PROJ):
    log.write(f"\n$ {cmd}\n")
    log.flush()
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"[exit {p.returncode} after {time.time() - t0:.0f}s]\n")
    log.flush()
    return p.returncode


def setup():
    os.makedirs(PROJ, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    # Kaggle auto-extracts uploaded archives, so the dataset usually holds the tree itself
    found = glob.glob("/kaggle/input/**/aerojev/engine.py", recursive=True)
    if found:
        src = os.path.dirname(os.path.dirname(found[0]))
        shutil.copytree(src, PROJ, dirs_exist_ok=True)
    else:
        bundle = glob.glob("/kaggle/input/**/aerojev_bundle.tar.gz", recursive=True)[0]
        with tarfile.open(bundle) as t:
            t.extractall(PROJ)
    with open(os.path.join(OUT, "setup.log"), "w") as log:
        sh("nvidia-smi", log)
        sh(f"{sys.executable} -m pip install -q 'transformers==5.20.2' 'peft==0.21.2' accelerate sentencepiece "
           "sentence-transformers scikit-learn", log)
        # the image's torchao 0.10 makes peft>=0.21 refuse to build LoRA layers; it is not needed here
        sh(f"{sys.executable} -m pip uninstall -y -q torchao", log)
        if any(j.get("needs_llama") for q in JOBS.values() for j in q):
            sh(f"{sys.executable} -m pip install -q llama-cpp-python==0.3.16 "
               "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu", log)
        sh("lscpu | head -20; nproc; free -g", log)
        sh(f"{sys.executable} -c 'import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__,torch.cuda.device_count())'", log)


def run_job(job, log, env):
    py = sys.executable
    k = job["kind"]
    if k == "eval":
        args = f"--model {job['model']} --sets {job.get('sets', 'val,test,gold,injection')} --out {OUT}/{job['out']} " \
               f"--dtype {job.get('dtype', 'float32')}"
        if job.get("adapter_input"):  # adapter produced by another kernel, mounted under /kaggle/input
            hits = glob.glob(f"/kaggle/input/**/{job['adapter_input']}/adapter_model.safetensors", recursive=True)
            if not hits:
                log.write(f"missing adapter {job['adapter_input']}\n")
                return 1
            args += f" --adapter {os.path.dirname(hits[0])}"
        elif job.get("adapter"):
            args += f" --adapter {OUT}/{job['adapter']}"
        if job.get("mode"):
            args += f" --mode {job['mode']}"
        if job.get("layout"):
            args += f" --layout {job['layout']}"
        if job.get("data"):
            args += f" --data {job['data']}"
        if job.get("limit"):
            args += f" --limit {job['limit']}"
        return sh(f"{py} -m eval.evaluate {args}", log, env)
    if k == "train":
        extra = " ".join(job.get("args", [])).replace("{OUT}", OUT)  # e.g. a teacher file written by an earlier job
        return sh(f"{py} -m train.train --model {job['model']} --out {OUT}/{job['out']} {extra}", log, env)
    if k == "export":
        with LLAMA_LOCK:  # first export job builds llama.cpp tools; the other GPU waits
            if not os.path.exists(f"{LLAMA}/build/bin/llama-quantize"):
                sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}", log)
                sh(f"{py} -m pip install -q {LLAMA}/gguf-py", log)
                sh(f"cmake -S {LLAMA} -B {LLAMA}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF "
                   f"&& cmake --build {LLAMA}/build --target llama-quantize -j 4", log)
        merged = f"/tmp/aerojev/merged-{job['name']}"
        gdir = f"{OUT}/gguf"
        os.makedirs(gdir, exist_ok=True)
        code = (
            "import torch,sys;from transformers import AutoModelForCausalLM,AutoTokenizer;from peft import PeftModel;"
            f"m=AutoModelForCausalLM.from_pretrained('{job['model']}',dtype=torch.float32);"
            f"m=PeftModel.from_pretrained(m,'{OUT}/{job['adapter']}').merge_and_unload();"
            f"m.save_pretrained('{merged}');AutoTokenizer.from_pretrained('{job['model']}').save_pretrained('{merged}')"
        )
        rc = sh(f"{py} -c \"{code}\"", log, env)
        f16 = f"{gdir}/{job['name']}-f16.gguf"
        rc |= sh(f"{py} {LLAMA}/convert_hf_to_gguf.py {merged} --outtype f16 --outfile {f16}", log, env)
        rc |= sh(f"{py} {LLAMA}/convert_hf_to_gguf.py {merged} --outtype q8_0 "
                 f"--outfile {gdir}/{job['name']}-q8_0.gguf", log, env)
        qbin = f"{LLAMA}/build/bin/llama-quantize"
        if os.path.exists(qbin):
            rc |= sh(f"{qbin} {f16} {gdir}/{job['name']}-q4_k_m.gguf Q4_K_M", log, env)
        if not job.get("keep_f16"):
            os.remove(f16) if os.path.exists(f16) else None
        shutil.rmtree(merged, ignore_errors=True)
        return rc
    if k == "cmd":  # free-form command; {OUT} and {PY} are substituted, {GGUF:<name>} finds an input file
        cmd = job["cmd"].replace("{OUT}", OUT).replace("{PY}", py).replace("{PROJ}", PROJ)
        for name in set(__import__("re").findall(r"\{GGUF:([^}]+)\}", cmd)):
            hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
            if not hits:
                log.write(f"missing input {name}\n")
                return 1
            cmd = cmd.replace("{GGUF:" + name + "}", hits[0])
        return sh(cmd, log, env)
    raise ValueError(k)


def worker(gpu, jobs, status):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=PROJ, TOKENIZERS_PARALLELISM="false",
               HF_HUB_DISABLE_PROGRESS_BARS="1")
    with open(os.path.join(OUT, f"gpu{gpu}.log"), "w") as log:
        for job in jobs:
            t0 = time.time()
            rc = run_job(job, log, env)
            status.append({"gpu": gpu, "job": job, "rc": rc, "seconds": round(time.time() - t0)})
            json.dump(status, open(os.path.join(OUT, "status.json"), "w"), indent=1)
            if rc != 0 and job.get("required"):
                break


def main():
    setup()
    status = []
    threads = [threading.Thread(target=worker, args=(int(g), q, status)) for g, q in JOBS.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
