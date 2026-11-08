#!/usr/bin/env python3
"""
run_all.py : unattended driver for TamperBench (wraps tamperbench.py; does NOT change what it computes).

WHAT IT DOES (in this order; each job is skipped if already finished, so re-running the SAME command resumes):
  0. environment checks (GPU, model files in local cache, code hash)
  1. JOB qwen2.5-7b seed 0 (main):  lmcheck -> pilot generate (gate) -> generate to n_main -> auto-size
     (grow n until >= val_target clean validation logs) -> tamper -> detect -> evaluate -> pack
  2. JOB swarm simulation (no LM, CPU, minutes)
  3. JOB llama3.1-8b seed 0 (cross-model replicate, small n)           [skipped if weights not cached]
  4. JOB qwen2.5-7b extra seeds (default: none; add with --seeds-a 0,1)
Results are stored per run under  runs/<model-tag>_seed<S>/  and summarised in  results_bundle/ :
  registry.csv / registry.jsonl      one row per run: model, seed, n, counts, timings, status, code hash
  combined/*.csv                     every results_*.csv of every run stacked, with run_name/model/seed columns
  RESULTS_INDEX.md                   human-readable index + headline AUROC table per run
  pack_<run>.zip                     small per-run package (results, figs, episodes.jsonl, logs)
  ALL_RESULTS_<timestamp>.zip (in the project root)  <- single archive with everything above

SAFETY RAILS: hard USD cap (price/h x elapsed; kills the running stage and stops; everything is resumable),
optional deadline, `touch STOP` for a graceful stop, retries for transient failures, a job that fails never blocks
the next job, the bundle is rebuilt every hour and on exit. The driver never shuts the machine down and never
uploads anything.

TYPICAL USE (inside tmux, in the folder that holds tamperbench.py):
  nohup python run_all.py --price <USD per hour> --cap-usd <limit> > driver.log 2>&1 &
  tail -f driver.log        # progress;   touch STOP   # graceful stop;   python run_all.py --status
DRY RUN OF THE PLAN (no GPU needed):  python run_all.py --plan-only
PIPELINE DRY RUN WITH A FAKE LM (no GPU, tiny sizes): python run_all.py --backend fake --n-pilot 2 --n-main 2 --n-max 2 --n-b 2
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="TamperBench unattended driver")
ap.add_argument("--script", default=str(HERE / "tamperbench.py"))
ap.add_argument("--root", default=str(HERE), help="project folder (runs/, results_bundle/, logs/ live here)")
ap.add_argument("--backend", choices=["hf", "fake"], default="hf", help="fake = pipeline dry run, no GPU")
ap.add_argument("--device", default="auto")
# models
ap.add_argument("--model-a", default="Qwen/Qwen2.5-7B-Instruct", help="main agent model")
ap.add_argument("--model-b", default="meta-llama/Llama-3.1-8B-Instruct", help="cross-model replicate")
ap.add_argument("--no-llama", action="store_true")
ap.add_argument("--no-swarm", action="store_true")
ap.add_argument("--seeds-a", default="0", help="generation seeds for model A, e.g. 0,1 (first one is the main run)")
ap.add_argument("--seeds-b", default="0")
# sizes
ap.add_argument("--n-pilot", type=int, default=6)
ap.add_argument("--n-main", type=int, default=80, help="starting n-per-family for the main run (auto-sizing may grow it up to --n-max)")
ap.add_argument("--n-max", type=int, default=160, help="auto-sizing never exceeds this")
ap.add_argument("--n-step", type=int, default=20)
ap.add_argument("--val-target", type=int, default=110, help="want >= this many clean validation logs (>=100 needed for TPR@1%%FPR)")
ap.add_argument("--n-b", type=int, default=12, help="n-per-family for the replicate model (small: AUROC only)")
ap.add_argument("--swarm-episodes", type=int, default=200)
# tamperbench settings
ap.add_argument("--agent", default="hybrid")
ap.add_argument("--max-steps", type=int, default=14)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--rewrite-with-lm", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--l3-n", type=int, default=8)
ap.add_argument("--detectors", default="b1,b2,b3,d1,d1abl,u1,d2,d3,v0")
ap.add_argument("--probe-layers", default="0.5")
ap.add_argument("--b2-max", type=int, default=1500)
ap.add_argument("--n-boot", type=int, default=500)
ap.add_argument("--ood-family", default="text_transform,drop_header")
# gates and budget
ap.add_argument("--min-valid-rate", type=float, default=0.30, help="abort a job if fewer valid episodes than this (saves money)")
ap.add_argument("--no-gates", action="store_true")
ap.add_argument("--price", type=float, default=0.59, help="machine price in USD per hour")
ap.add_argument("--spent-usd", type=float, default=0.0, help="USD already spent BEFORE this driver started")
ap.add_argument("--cap-usd", type=float, default=18.0, help="hard cap in USD")
ap.add_argument("--deadline", default="", help="ISO time WITH offset, e.g. 2026-11-08T17:00:00+01:00; optional jobs that would end later are skipped")
ap.add_argument("--retries", type=int, default=2)
ap.add_argument("--pack-all-data", action="store_true", help="also put dataset/scores/pkl in the packs (can be large)")
ap.add_argument("--skip-lmcheck", action="store_true")
ap.add_argument("--reset-job", default="", help="forget driver progress marks of this run name (data is kept; tamperbench resumes)")
ap.add_argument("--plan-only", action="store_true")
ap.add_argument("--status", action="store_true")
A = ap.parse_args()

ROOT = Path(A.root).resolve()
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
BUNDLE = ROOT / "results_bundle"
STATE_PATH = ROOT / "driver_state.json"
STOP_FILE = ROOT / "STOP"
for d in (RUNS, LOGS, BUNDLE):
    d.mkdir(parents=True, exist_ok=True)

import tamperbench as tb  # noqa: E402  (helpers only: read_jsonl, assign_split; importing runs no experiment)

T0 = time.time()


# ----------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------
def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def say(msg: str) -> None:
    print(f"[driver {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha_file(p: Path) -> str:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except Exception:
        return "?"


def short_tag(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-instruct", "").replace("_", "-")


def spent() -> float:
    return A.spent_usd + (time.time() - T0) / 3600.0 * A.price


def deadline_dt():
    if not A.deadline:
        return None
    dt = datetime.datetime.fromisoformat(A.deadline)
    if dt.tzinfo is None:
        sys.exit("--deadline needs an explicit UTC offset, e.g. 2026-11-08T17:00:00+01:00")
    return dt


class StopDriver(Exception):
    pass


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"jobs": {}, "lmcheck": {}, "est_hours_per_episode": None, "n_final_a": None}


STATE = load_state()


def save_state() -> None:
    atomic_write(STATE_PATH, json.dumps(STATE, indent=2, default=str))


def job_state(name: str) -> dict:
    return STATE["jobs"].setdefault(name, {"status": "new", "stages": {}, "seconds": 0.0})


# ----------------------------------------------------------------------------------------------
# running tamperbench.py subcommands with a watchdog
# ----------------------------------------------------------------------------------------------
class Cur:
    proc = None
    reason = None  # "budget" | "signal" | "stopfile"


def kill_current(sig=signal.SIGTERM):
    p = Cur.proc
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, sig)
        except Exception:
            pass


def _on_signal(signum, _frm):
    Cur.reason = "signal"
    say(f"signal {signum} received: stopping the running stage (everything is resumable)")
    kill_current()


def watchdog():
    last_bundle = time.time()
    while True:
        time.sleep(15)
        if STOP_FILE.exists() and Cur.reason is None:
            Cur.reason = "stopfile"
            say("STOP file found: stopping the running stage gracefully")
            kill_current()
        if spent() >= A.cap_usd and Cur.reason is None:
            Cur.reason = "budget"
            say(f"BUDGET CAP reached (~${spent():.2f} >= ${A.cap_usd:.2f}): stopping the running stage")
            kill_current()
        if Cur.reason is not None and Cur.proc is not None and Cur.proc.poll() is None:
            time.sleep(45)
            kill_current(signal.SIGKILL)
        if time.time() - last_bundle > 3600:
            last_bundle = time.time()
            try:
                say("hourly backup: rebuilding results bundle")
                build_bundle(quiet=True)
            except Exception as e:
                say(f"hourly backup failed (ignored): {e}")


def run_cmd(args_list, logfile: Path, label: str):
    cmd = [sys.executable, A.script] + [str(x) for x in args_list]
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    tail = collections.deque(maxlen=80)
    with open(logfile, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {now()} | {label}\n===== {' '.join(cmd)}\n")
        lf.flush()
        say(f"{label}  ->  log: {logfile.relative_to(ROOT)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env, cwd=str(ROOT), start_new_session=True)
        Cur.proc = proc
        try:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                sys.stdout.write(line)
                tail.append(line)
        finally:
            rc = proc.wait()
            Cur.proc = None
    if Cur.reason is not None:
        raise StopDriver(Cur.reason)
    return rc, list(tail)


def stage(job_name: str, key: str, args_list, label: str, log_stem: str, retries=None):
    """Run one tamperbench stage with retries; mark it done in driver_state. Returns the output tail."""
    js = job_state(job_name)
    if js["stages"].get(key, {}).get("ok"):
        say(f"{label}: already done ({key}), skipping")
        return js["stages"][key].get("tail", [])
    retries = A.retries if retries is None else retries
    t0 = time.time()
    for attempt in range(retries + 1):
        rc, tail = run_cmd(args_list, LOGS / f"{log_stem}.log", f"{label} (attempt {attempt + 1}/{retries + 1})")
        text = "".join(tail)
        if rc == 0:
            js["stages"][key] = {"ok": True, "seconds": round(time.time() - t0, 1), "ended": now(), "tail": tail[-15:]}
            js["seconds"] = js.get("seconds", 0.0) + (time.time() - t0)
            save_state()
            return tail
        if "REFUSING to resume" in text:
            raise RuntimeError("config guard refused to resume this run-dir (model/seed/temperature differ). "
                               "Use a new run dir or inspect runs/*/config.json.")
        say(f"{label} failed with exit code {rc}; last lines:\n" + "".join(tail[-8:]))
        if attempt < retries:
            time.sleep(20)
    raise RuntimeError(f"{label} failed after {retries + 1} attempts (see {LOGS / (log_stem + '.log')})")


# ----------------------------------------------------------------------------------------------
# environment checks
# ----------------------------------------------------------------------------------------------
def gpu_ok() -> bool:
    if A.backend == "fake":
        return True
    try:
        out = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        return out.endswith("True")
    except Exception:
        return False


def model_cached(model_id: str) -> bool:
    """True if the model config is in the local HF cache; if online, the child process may download it."""
    if A.backend == "fake":
        return True
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(model_id, "config.json")
        cached = isinstance(p, str)
    except Exception:
        cached = False
    return cached or not offline


# ----------------------------------------------------------------------------------------------
# diagnostics on episodes (mirror of cmd_tamper's validity filter)
# ----------------------------------------------------------------------------------------------
def is_valid(e) -> bool:
    return bool(e["success"]) and (e["condition"] == "clean" or bool(e["violation_done"])) and len(e["steps"]) >= 3 \
        and (e["condition"] == "violation" or not e["violation_steps"])


def diagnose(rd: Path):
    eps = tb.read_jsonl(rd / "episodes.jsonl")
    ood = set(A.ood_family.split(","))
    rows = collections.OrderedDict()
    for e in eps:
        k = (e["family"], e["condition"])
        r = rows.setdefault(k, dict(n=0, success=0, valid=0, llm_pure=0))
        r["n"] += 1
        r["success"] += bool(e["success"])
        r["valid"] += is_valid(e)
        r["llm_pure"] += bool(e.get("llm_pure"))
    n_all = len(eps)
    n_valid = sum(is_valid(e) for e in eps)
    val_est = sum(1 for e in eps if is_valid(e) and e["family"] not in ood
                  and tb.assign_split(e["family"], e["seed"], A.ood_family) == "val")
    md = ["| family | condition | episodes | task success | valid | llm_pure |", "|---|---|---:|---:|---:|---:|"]
    for (f, c), r in sorted(rows.items()):
        md.append(f"| {f} | {c} | {r['n']} | {r['success']} | {r['valid']} | {r['llm_pure']} |")
    info = dict(episodes=n_all, valid=n_valid, valid_rate=(n_valid / n_all if n_all else 0.0),
                val_neg_est=val_est, table_md="\n".join(md),
                llm_pure_rate=(sum(bool(e.get("llm_pure")) for e in eps) / n_all if n_all else 0.0))
    return info


def write_pilot_report(rd: Path, tag: str, n: int, info: dict) -> None:
    txt = (f"# Generation report ({tag}, n-per-family={n}) - {now()}\n\n"
           f"episodes={info['episodes']} valid={info['valid']} valid_rate={info['valid_rate']:.2f} "
           f"llm_pure_rate={info['llm_pure_rate']:.2f} est_clean_val_logs={info['val_neg_est']}\n\n{info['table_md']}\n")
    atomic_write(rd / f"gen_report_n{n}.md", txt)
    print(txt, flush=True)


def val_neg_dataset(rd: Path) -> int:
    rows = tb.read_jsonl(rd / "dataset.jsonl")
    return sum(1 for r in rows if r["split"] == "val" and not r["tampered"])


# ----------------------------------------------------------------------------------------------
# one LLM job (generate -> tamper -> detect -> evaluate)
# ----------------------------------------------------------------------------------------------
def common(rd: Path, model: str, seed: int):
    return ["--run-dir", rd, "--backend", A.backend, "--model", model, "--device", A.device, "--seed", seed]


def gen_args(n: int):
    return ["--agent", A.agent, "--n-per-family", n, "--max-steps", A.max_steps, "--temperature", A.temperature]


def tam_args():
    a = ["--l3-n", A.l3_n, "--ood-family", A.ood_family]
    if A.rewrite_with_lm:
        a.append("--rewrite-with-lm")
    return a


def det_args():
    return ["--detectors", A.detectors, "--probe-layers", A.probe_layers, "--b2-max", A.b2_max]


def grow_n(n: int, est: int) -> int:
    want = math.ceil(n * A.val_target / max(est, 1) * 1.1 / A.n_step) * A.n_step
    return int(min(A.n_max, max(n + A.n_step, want)))


def run_llm_job(job: dict) -> None:
    name, model, seed, sizing = job["name"], job["model"], job["seed"], job["sizing"]
    rd = RUNS / name
    js = job_state(name)
    js.update(status="running", model=model, seed=seed, started=js.get("started") or now())
    save_state()
    cm = lambda: common(rd, model, seed)

    if not A.skip_lmcheck and not STATE["lmcheck"].get(model):
        tail = stage(name, "lmcheck", ["lmcheck"] + cm(), f"{name}: lmcheck", f"{name}_lmcheck", retries=0)
        txt = "".join(tail)
        bad = ("WARNING" in txt) or ("parsed action: None" in txt)
        if bad and A.backend != "fake":
            js["stages"].pop("lmcheck", None)
            raise RuntimeError("lmcheck failed (invalid JSON action or NLL direction wrong): not spending GPU time. See logs.")
        STATE["lmcheck"][model] = True
        save_state()

    # pilot (main job only): cheap first look + gate before the long run
    n_first = A.n_pilot if sizing else job["n"]
    stage(name, f"generate@{n_first}", ["generate"] + cm() + gen_args(n_first), f"{name}: generate n={n_first}", f"{name}_generate")
    info = diagnose(rd)
    write_pilot_report(rd, name, n_first, info)
    if not A.no_gates and info["valid_rate"] < A.min_valid_rate:
        js["status"] = "aborted_low_valid_rate"
        js["note"] = f"valid rate {info['valid_rate']:.2f} < {A.min_valid_rate} after n={n_first}"
        save_state()
        raise RuntimeError(js["note"] + " - agent too weak or unreliable; fix before spending more.")

    n = job["n"]
    if sizing and n != n_first:
        stage(name, f"generate@{n}", ["generate"] + cm() + gen_args(n), f"{name}: generate n={n}", f"{name}_generate")
        info = diagnose(rd)
        write_pilot_report(rd, name, n, info)
        if not A.no_gates and info["valid_rate"] < A.min_valid_rate:
            js["status"] = "aborted_low_valid_rate"
            save_state()
            raise RuntimeError(f"valid rate {info['valid_rate']:.2f} < {A.min_valid_rate}")
    # auto-sizing from the episodes (cheap estimate of clean validation logs)
    while sizing and info["val_neg_est"] < A.val_target and n < A.n_max:
        n = grow_n(n, info["val_neg_est"])
        say(f"{name}: only ~{info['val_neg_est']} clean validation logs -> growing n-per-family to {n}")
        stage(name, f"generate@{n}", ["generate"] + cm() + gen_args(n), f"{name}: generate n={n}", f"{name}_generate")
        info = diagnose(rd)
        write_pilot_report(rd, name, n, info)
    js["n_per_family"] = n
    save_state()
    safe_bundle(f"after generate ({name})")

    stage(name, f"tamper@{n}", ["tamper"] + cm() + tam_args(), f"{name}: tamper", f"{name}_tamper")
    actual = val_neg_dataset(rd)
    say(f"{name}: clean validation logs in dataset = {actual} (need >= 100 for TPR@1%FPR)")
    if sizing and actual < 100 and n < A.n_max and not js.get("regrown"):
        js["regrown"] = True
        n = int(min(A.n_max, max(n + A.n_step, grow_n(n, actual))))
        say(f"{name}: too few clean validation logs -> one regrowth to n={n} (tamper cache is rebuilt)")
        stage(name, f"generate@{n}", ["generate"] + cm() + gen_args(n), f"{name}: generate n={n}", f"{name}_generate")
        js["n_per_family"] = n
        stage(name, f"tamper@{n}", ["tamper"] + cm() + tam_args(), f"{name}: tamper", f"{name}_tamper")
        actual = val_neg_dataset(rd)
    js["val_neg_dataset"] = actual
    save_state()

    stage(name, f"detect@{n}", ["detect"] + cm() + det_args(), f"{name}: detect", f"{name}_detect")
    safe_bundle(f"after detect ({name})")
    stage(name, f"evaluate@{n}", ["evaluate"] + cm() + ["--n-boot", A.n_boot], f"{name}: evaluate", f"{name}_evaluate")
    js["status"] = "done"
    js["ended"] = now()
    save_state()


def run_swarm_job(job: dict) -> None:
    name = job["name"]
    rd = RUNS / name
    js = job_state(name)
    js.update(status="running", model="(no LM: scripted simulation)", seed="-", started=js.get("started") or now())
    save_state()
    stage(name, "swarm", ["swarm", "--run-dir", rd, "--n-episodes", A.swarm_episodes], f"{name}: swarm simulation", f"{name}_swarm")
    js["status"] = "done"
    js["ended"] = now()
    save_state()


# ----------------------------------------------------------------------------------------------
# result collection: registry, combined CSVs, index, bundle
# ----------------------------------------------------------------------------------------------
def run_info(name: str) -> dict:
    rd = RUNS / name
    js = STATE["jobs"].get(name, {})
    info = dict(run_name=name, model=js.get("model", ""), seed=js.get("seed", ""), status=js.get("status", "new"),
                n_per_family=js.get("n_per_family", ""), hours=round(js.get("seconds", 0.0) / 3600, 2),
                started=js.get("started", ""), ended=js.get("ended", ""), note=js.get("note", ""))
    if (rd / "episodes.jsonl").exists():
        d = diagnose(rd)
        info.update(episodes=d["episodes"], valid=d["valid"], valid_rate=round(d["valid_rate"], 3),
                    llm_pure_rate=round(d["llm_pure_rate"], 3))
    if (rd / "dataset.jsonl").exists():
        rows = tb.read_jsonl(rd / "dataset.jsonl")
        info.update(variants=len(rows), clean_val_logs=sum(1 for r in rows if r["split"] == "val" and not r["tampered"]))
    info["errors_log_lines"] = len((rd / "errors.log").read_text().splitlines()) if (rd / "errors.log").exists() else 0
    info["code_sha_tamperbench"] = sha_file(Path(A.script))
    info["code_sha_driver"] = sha_file(Path(__file__))
    return info


def read_csv_rows(p: Path):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def headline_md(name: str) -> str:
    p = RUNS / name / "results_main.csv"
    if not p.exists():
        return ""
    rows = read_csv_rows(p)
    tts = {r["tamper_type"] for r in rows}
    tt = "T1-T4" if "T1-T4" in tts else ("all" if "all" in tts else sorted(tts)[0])
    sel = [r for r in rows if r["split"] == "test_id" and r["tamper_type"] == tt]
    dets = sorted({r["detector"] for r in sel})
    levels = sorted({r["level"] for r in sel}, key=lambda x: int(float(x)))

    def f(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)
    out = [f"**{name}**: test_id, tamper_type={tt}, AUROC (95% CI) per adversary level",
           "", "| detector | " + " | ".join(f"L{int(float(l))}" for l in levels) + " |", "|---|" + "---:|" * len(levels)]
    for d in dets:
        cells = []
        for l in levels:
            m = [r for r in sel if r["detector"] == d and r["level"] == l]
            cells.append(f"{f(m[0]['auroc'])} ({f(m[0]['ci_lo'])}-{f(m[0]['ci_hi'])})" if m else "-")
        out.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_bundle(quiet=False) -> Path:
    names = [d.name for d in sorted(RUNS.iterdir()) if d.is_dir()]
    infos = [run_info(n) for n in names]
    # registry
    cols = sorted({k for i in infos for k in i}, key=lambda k: (k != "run_name", k))
    with open(BUNDLE / "registry.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(infos)
    atomic_write(BUNDLE / "registry.jsonl", "\n".join(json.dumps(i) for i in infos) + "\n")
    # combined CSVs
    comb_dir = BUNDLE / "combined"
    comb_dir.mkdir(exist_ok=True)
    files = sorted({p.name for n in names for p in (RUNS / n).glob("results_*.csv")})
    for fn in files:
        allrows, keys = [], []
        for n in names:
            p = RUNS / n / fn
            if not p.exists():
                continue
            js = STATE["jobs"].get(n, {})
            for r in read_csv_rows(p):
                r = dict(run_name=n, model=js.get("model", ""), seed=js.get("seed", ""), **r)
                allrows.append(r)
                keys += [k for k in r if k not in keys]
        if allrows:
            with open(comb_dir / fn.replace("results_", "all_runs_"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(allrows)
    # index
    md = [f"# TamperBench results index ({now()})", "",
          f"Driver spend estimate this session: ~${spent() - A.spent_usd:.2f} (+ ${A.spent_usd:.2f} reported before). "
          "All numbers below come from tamperbench.py output.", "",
          "| run | model | seed | status | n/family | episodes | valid | variants | clean val logs | hours |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for i in infos:
        md.append(f"| {i['run_name']} | {i['model']} | {i['seed']} | {i['status']} | {i['n_per_family']} | {i.get('episodes', '')} | "
                  f"{i.get('valid', '')} | {i.get('variants', '')} | {i.get('clean_val_logs', '')} | {i['hours']} |")
    for n in names:
        h = headline_md(n)
        if h:
            md += ["", h]
    atomic_write(BUNDLE / "RESULTS_INDEX.md", "\n".join(md) + "\n")
    # per-run packs (tamperbench's own `pack`, run inside the bundle folder)
    for n in names:
        logs = ",".join(str(p) for p in sorted(LOGS.glob(f"{n}_*.log")))
        cmd = [sys.executable, A.script, "pack", "--run-dir", str(RUNS / n), "--with-episodes"]
        if logs:
            cmd += ["--logs", logs]
        if A.pack_all_data:
            cmd.append("--all-data")
        subprocess.run(cmd, cwd=str(BUNDLE), capture_output=True, text=True)
    # driver files
    for p in (STATE_PATH, Path(A.script), Path(__file__), ROOT / "driver.log"):
        if p.exists():
            try:
                (BUNDLE / "driver_files").mkdir(exist_ok=True)
                (BUNDLE / "driver_files" / p.name).write_bytes(p.read_bytes())
            except Exception:
                pass
    # one final zip (keep only the newest)
    for old in ROOT.glob("ALL_RESULTS_*.zip"):
        old.unlink()
    out = ROOT / f"ALL_RESULTS_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUNDLE.rglob("*")):
            if p.is_file():
                z.write(p, str(Path("results_bundle") / p.relative_to(BUNDLE)))
        for p in sorted(LOGS.glob("*.log")):
            z.write(p, str(Path("logs") / p.name))
    if not quiet:
        say(f"BUNDLE READY: {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def safe_bundle(why: str) -> None:
    try:
        build_bundle(quiet=True)
        say(f"backup refreshed ({why})")
    except Exception as e:
        say(f"backup failed (ignored): {e}")


# ----------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------
def make_plan():
    seeds_a = [int(s) for s in A.seeds_a.split(",") if s != ""]
    seeds_b = [int(s) for s in A.seeds_b.split(",") if s != ""]
    ta, tb_ = short_tag(A.model_a), short_tag(A.model_b)
    plan = [dict(kind="llm", name=f"{ta}_seed{seeds_a[0]}", model=A.model_a, seed=seeds_a[0], sizing=True, n=A.n_main, optional=False)]
    if not A.no_swarm:
        plan.append(dict(kind="swarm", name="swarm", optional=False))
    if not A.no_llama:
        for s in seeds_b:
            plan.append(dict(kind="llm", name=f"{tb_}_seed{s}", model=A.model_b, seed=s, sizing=False, n=A.n_b, optional=True))
    for s in seeds_a[1:]:
        plan.append(dict(kind="llm", name=f"{ta}_seed{s}", model=A.model_a, seed=s, sizing=False, n=None, optional=True))
    return plan


def est_hours(job) -> float:
    n = job.get("n") or STATE.get("n_final_a") or A.n_main
    per_ep = STATE.get("est_hours_per_episode") or 0.0125
    return per_ep * n * len(tb.FAMILIES) * 2


def can_start_optional(job) -> tuple[bool, str]:
    eh = est_hours(job)
    if spent() + eh * A.price > A.cap_usd * 0.95:
        return False, f"estimated cost ~${eh * A.price:.2f} would exceed the cap (spent ~${spent():.2f} of ${A.cap_usd})"
    dl = deadline_dt()
    if dl is not None:
        end = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=eh)
        if end > dl:
            return False, f"estimated end {end.isoformat(timespec='minutes')} is after the deadline {dl.isoformat(timespec='minutes')}"
    return True, ""


def print_plan(plan):
    say("PLAN (run order):")
    for i, j in enumerate(plan, 1):
        extra = "" if j["kind"] == "swarm" else f" model={j['model']} seed={j['seed']} n/family={j['n'] or 'same as main'}" + (" (auto-sized)" if j.get("sizing") else "")
        say(f"  {i}. {j['name']}  [{j['kind']}]{extra}  {'(optional, budget/deadline-gated)' if j.get('optional') else ''}")


def main():
    plan = make_plan()
    if A.reset_job:
        STATE["jobs"].pop(A.reset_job, None)
        save_state()
        say(f"progress marks of {A.reset_job} cleared (its data in runs/{A.reset_job} is kept; tamperbench resumes from it)")
    if A.status:
        for j in plan:
            js = STATE["jobs"].get(j["name"], {})
            print(f"{j['name']:<28} status={js.get('status', 'new'):<24} stages_done={[k for k, v in js.get('stages', {}).items() if v.get('ok')]}")
        print(f"spent (reported + driver) is only known while the driver runs; state file: {STATE_PATH}")
        return
    print_plan(plan)
    if A.plan_only:
        return
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    if not gpu_ok():
        sys.exit("[driver] no CUDA GPU visible to torch: refusing to start (a 7B model on CPU would burn hours). Use --backend fake for a dry run.")
    threading.Thread(target=watchdog, daemon=True).start()
    say(f"code hashes: tamperbench={sha_file(Path(A.script))} driver={sha_file(Path(__file__))}; cap ${A.cap_usd}; price ${A.price}/h; already spent ${A.spent_usd}")
    stopped = None
    try:
        for job in plan:
            js = job_state(job["name"])
            if js["status"] == "done":
                say(f"{job['name']}: already done, skipping")
                continue
            if job["kind"] == "llm" and not model_cached(job["model"]):
                js.update(status="skipped_model_not_cached", note="weights not in local HF cache and HF_HUB_OFFLINE=1")
                save_state()
                say(f"{job['name']}: SKIPPED - {job['model']} is not in the local cache (download it first, then re-run the same command)")
                continue
            if job.get("optional"):
                ok, why = can_start_optional(job)
                if not ok:
                    js.update(status="skipped_budget_or_deadline", note=why)
                    save_state()
                    say(f"{job['name']}: SKIPPED - {why}")
                    continue
            if job["kind"] == "llm" and job["n"] is None:
                job = dict(job, n=STATE.get("n_final_a") or A.n_main)
            if spent() >= A.cap_usd:
                raise StopDriver("budget")
            t_job = time.time()
            try:
                run_swarm_job(job) if job["kind"] == "swarm" else run_llm_job(job)
            except StopDriver:
                raise
            except Exception as e:
                js = job_state(job["name"])
                if js.get("status") == "running":
                    js["status"] = "failed"
                js["note"] = f"{type(e).__name__}: {e}"[:500]
                save_state()
                say(f"JOB {job['name']} FAILED: {e}  -> continuing with the next job")
                continue
            if job["kind"] == "llm" and job.get("sizing"):
                STATE["n_final_a"] = job_state(job["name"]).get("n_per_family")
            if job["kind"] == "llm" and STATE.get("est_hours_per_episode") is None:
                nn = job_state(job["name"]).get("n_per_family") or job["n"]
                STATE["est_hours_per_episode"] = (time.time() - t_job) / 3600.0 / max(1, nn * len(tb.FAMILIES) * 2)
            save_state()
            say(f"JOB {job['name']} DONE in {(time.time() - t_job) / 3600:.2f} h; spend estimate ~${spent():.2f}")
            safe_bundle(f"job {job['name']} finished")
    except StopDriver as s:
        stopped = str(s)
        for j in STATE["jobs"].values():
            if j.get("status") == "running":
                j["status"] = f"stopped_{stopped}"
        save_state()
        say(f"DRIVER STOPPED: reason = {stopped}. Re-run the same command later to resume (all stages are resumable).")
    finally:
        try:
            build_bundle()
        except Exception as e:
            say(f"final bundle failed: {e}")
        say("The driver never shuts the machine down. Download ALL_RESULTS_*.zip before stopping it.")
        say(f"spend estimate for this session ~${spent() - A.spent_usd:.2f}; cumulative incl. reported ~${spent():.2f} of cap ${A.cap_usd}")


if __name__ == "__main__":
    main()
