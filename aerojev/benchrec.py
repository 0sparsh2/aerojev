"""One latency-record schema for every system (LLM readout, generation, classical)."""

import json
import os
import platform
import resource
import statistics
import subprocess
import sys


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def host_label():
    name = cpu_name()
    if sys.platform == "darwin":
        return f"mac ({name})"
    return f"x86 ({name}, {os.cpu_count()} vCPU)"


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1e6 if sys.platform == "darwin" else rss / 1e3)  # bytes on macOS, KiB on Linux


def record(system, engine, latencies, n_questions, threads, **extra):
    lat = sorted(latencies)
    mean = statistics.mean(lat)
    rec = {
        "host": host_label(), "system": system, "engine": engine, "threads": threads,
        "n_questions": n_questions, "n_records": len(lat),
        "mean_s": round(mean, 4), "p50_s": round(lat[len(lat) // 2], 4),
        "p95_s": round(lat[max(0, int(len(lat) * 0.95) - 1)], 4),
        "ms_per_decision": round(1000 * mean / n_questions, 2), "peak_rss_mb": peak_rss_mb(),
    }
    rec.update(extra)
    return rec


def append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
