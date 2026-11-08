# TamperBench

Post-hoc forensic detection of tampered LLM-agent traces: a benchmark, a detector suite,
and an analysis of which trusted signals help most. Built for the Swarm Alignment Hackathon
(Apart Research x AI Alignment Foundation, Nov 2026), Track 4: Multi-Agent Oversight and Control.

## What this is
When an agent's stored log may have been edited (deletion, fabrication, tool-call spoofing,
observation falsification, LLM rewriting), how well can the edit be detected and localized
afterwards, and which cheap trusted signals buy the most detection? TamperBench provides:
- a toy sandbox ("SandboxShell-Lite", 8 task families, synthetic canary side-action),
- five tamper operators (T1-T5) and four adversary levels (L0-L3),
- a 2x2 design (clean/violation x untampered/tampered) that removes the "tampering = misbehaviour" shortcut,
- detectors: B1-B3 baselines, D1 host-counter consistency, D2 divergence, D3 probe, D4 fusion, U1 replay oracle,
- a scripted swarm simulation of cross-log "witness" detection.

Everything is synthetic and runs in a throwaway temp directory. No network, no real data,
no real credentials. Results are for a toy setting; see the report's limitations.

## Data
Generated data (dataset.jsonl, episodes.jsonl, scores.jsonl) is on Hugging Face: <HF_DATASET_LINK>.
Datasets are not bit-reproducible (wall-clock counters differ between runs), so we release the
generated episodes.

## Reproduce
Requirements: Python 3.10+, `pip install -r requirements.txt`, one GPU with >= 24 GB for the 7B models.
Sanity check without GPU:  `python tamperbench.py selftest`
Full run (what we used, RTX A6000 48 GB, ~5 GPU hours):
    nohup python run_all.py --spent-usd 0 --price 0.59 --cap-usd 18 > driver.log 2>&1 &
Every stage is resumable: re-run the same command after an interruption.
D3 control (L0/L1-only training): <EXACT COMMAND>
Models: Qwen/Qwen2.5-7B-Instruct (main), meta-llama/Llama-3.1-8B-Instruct (replicate, gated).
Code version: tamperbench.py sha256 = <HASH>.

## Results
Pre-computed tables and figures are in `results/`. Headline numbers are in `results/RESULTS_INDEX.md`
and the report (`report/`).

## Pre-existing code
<statement: see item 2>

## Dual-use note
The benchmark includes operators that rewrite logs to hide a benign synthetic action. They
operate only on synthetic logs from a toy sandbox. See Appendix A of the report.

## License
Code: MIT. Data: CC BY 4.0.

## Citation
Vahideh Zolfaghari, "TamperBench", Swarm Alignment Hackathon 2026.
