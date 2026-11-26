#!/usr/bin/env python3
"""TamperBench preflight: run on the machine (RunPod) BEFORE detect/evaluate. Changes nothing.
Usage: python preflight.py runs/qwen2.5-7b_seed0 [--model Qwen/Qwen2.5-7B-Instruct]"""
import sys, os, json, shutil, hashlib, subprocess, importlib
from pathlib import Path
rd = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "runs/default")
model = "Qwen/Qwen2.5-7B-Instruct"
if "--model" in sys.argv: model = sys.argv[sys.argv.index("--model") + 1]
ok = lambda m: print("  [ OK ]", m); warn = lambda m: print("  [WARN]", m); bad = lambda m: print("  [FAIL]", m)
mb = lambda p: f"{Path(p).stat().st_size/1e6:,.1f} MB"

print("1. Environment (U1 replay re-executes logged python: the SAME environment as generation is required)")
print("   python", sys.version.split()[0], "| expected 3.12.3 for the reported runs")
for m in ("numpy", "sklearn", "torch", "transformers", "matplotlib"):
    try: print(f"   {m} {getattr(importlib.import_module(m), '__version__', '?')}")
    except Exception: warn(f"{m} not installed")
for m in ("pandas", "scipy", "requests"):   # third-party modules the agent's run_python could import
    try: importlib.import_module(m); warn(f"{m} IS installed: logs made WITHOUT it will not replay identically (U1 false mismatches)")
    except Exception: ok(f"{m} not installed (matches the generation environment if it was absent then)")
try: os.fstat(0); ok("stdin (fd 0) is open") if os.isatty(0) is False else warn("stdin is a terminal")
except OSError: warn("stdin (fd 0) is CLOSED: this is how the reported runs behaved (agent code reading sys.stdin got 'Bad file descriptor')")
try:
    import torch
    if torch.cuda.is_available(): ok(f"GPU {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
    else: warn("no CUDA GPU: fine for detect/evaluate ONLY if all LM caches below are complete")
except Exception: warn("torch unavailable")
du = shutil.disk_usage(rd if rd.exists() else ".")
(ok if du.free > 20e9 else warn)(f"free disk {du.free/1e9:.0f} GB (hidden-state cache + model weights can be large)")

print(f"\n2. Run directory {rd}")
if not rd.exists(): bad("run directory not found"); sys.exit(1)
for f in ("config.json", "episodes.jsonl", "dataset.jsonl", "scores.jsonl", "tamper_cache.jsonl",
          "detect_cache_d2d3.pkl", "detect_cache_b2.jsonl", "manifest.jsonl", "errors.log"):
    p = rd / f; (ok if p.exists() else warn)(f"{f}: {mb(p) if p.exists() else 'MISSING'}")
if (rd / "config.json").exists():
    c = json.loads((rd / "config.json").read_text())
    print("   config:", {k: c.get(k) for k in ("backend", "model", "seed", "temperature", "max_steps", "n_per_family")})
    if c.get("model") != model: warn(f"config model {c.get('model')} != --model {model}: the D2/D3/B2 cache keys will NOT match")
if (rd / "episodes.jsonl").exists():
    n = sum(1 for _ in open(rd / "episodes.jsonl")); (ok if n == 1600 else warn)(f"{n} episodes (reported run: 1600)")
if (rd / "dataset.jsonl").exists():
    n = sum(1 for _ in open(rd / "dataset.jsonl")); (ok if n == 17694 else warn)(f"{n} variants (reported run: 17,694)")

print("\n3. Can detect run WITHOUT loading the LM? (needs both caches, same model + --probe-layers 0.5)")
need = [rd / "detect_cache_d2d3.pkl", rd / "detect_cache_b2.jsonl"]
if all(p.exists() and p.stat().st_size > 0 for p in need): ok("caches present: detect/evaluate can run on CPU, no model download")
else: warn("a cache is missing: detect must recompute D2/D3/B2 -> needs the GPU and ~1.3 h (reported: detection 4,530 s)")

print("\n4. Code")
for f in ("tamperbench.py", "run_all.py"):
    if Path(f).exists(): print(f"   {f} sha256 {hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]}")
print("   Record these hashes and `pip freeze` next to every result you report.")
