# TamperBench
The submitted version is release `v0.1-hackathon`.

Post-hoc forensic detection of tampered LLM-agent traces: a benchmark, a detector suite, and an
analysis of which trusted signals help most.

Built for the Swarm Alignment Hackathon (Apart Research x AI Alignment Foundation, November 2026),
Track 4: Multi-Agent Oversight and Control.

## Question

If an agent's stored log has been edited (deleted steps, fabricated steps, spoofed tool calls,
falsified observations, rewritten text) and no integrity layer protected it, how well can the edit
be detected and localized afterwards? Which cheap trusted signals help most?

## What is in this repository

- `tamperbench.py`: the whole pipeline in one file (sandbox, agent loop, tamper operators,
  detectors, evaluation, swarm simulation).
- `run_all.py`: unattended driver that runs the pipeline for several models and seeds.
- `results/`: result tables (CSV), summaries and figures of the runs reported in the paper,
  including the run manifests with code hashes.
- `report/`: the submitted report (PDF).

Large generated data (datasets, episodes, scores) is not stored here; see the Data section.

## What the benchmark contains

- A small sandbox (SandboxShell-Lite) with 8 task families and whitelisted tools.
- Five tamper types: T1 deletion, T2 fabricated steps, T3 tool-call spoofing,
  T4 observation falsification, T5 LLM rewrite.
- Four adversary levels: L0 naive, L1 format-aware, L2 content-consistent,
  L3 best-of-n against a surrogate detector.
- A 2x2 design (clean or violation run, untampered or tampered log), so a detector cannot score well
  just by flagging misbehaviour. A synthetic "canary" file stands in for forbidden behaviour.
- Detectors: B1 text/metadata classifier, B2 LLM judge, B3 one-class model, D1 host-counter
  consistency, D2 likelihood divergence, D3 activation probe, D4 fusion, U1 replay oracle,
  and a violation-only control.
- A scripted four-agent simulation of cross-log "witness" detection.

Everything is synthetic. No real data, credentials, or vulnerabilities are involved.

## Data

The generated datasets (`dataset.jsonl`, `episodes.jsonl`, `scores.jsonl` for each run) will be
released on Hugging Face: link to be added here.

The generated data is not bit-reproducible (wall-clock counters differ between runs), which is why
the generated episodes are released rather than only the code.

## Quick check (no GPU, no model download)

```
pip install -r requirements.txt
python tamperbench.py selftest
```

This runs the full pipeline with a fake language model on tiny sizes. The numbers it prints are
pipeline tests, not results. It should end with `SELFTEST OK`.

## Reproducing the reported results

Hardware used: one NVIDIA RTX A6000 (48 GB). Python 3.12.3, torch 2.8.0, transformers 5.22.2.
Models: `Qwen/Qwen2.5-7B-Instruct` (main) and `meta-llama/Llama-3.1-8B-Instruct`
(small replicate; gated, needs access on Hugging Face).

Stage commands for the main run (Qwen2.5-7B-Instruct, seed 0). Add
`--run-dir runs/qwen2.5-7b_seed0 --backend hf --model Qwen/Qwen2.5-7B-Instruct` to each command.

```
python tamperbench.py generate --agent hybrid --n-per-family 100 --max-steps 14 --temperature 0.7 --seed 0
python tamperbench.py tamper   --l3-n 8 --ood-family text_transform,drop_header --rewrite-with-lm
python tamperbench.py detect   --detectors b1,b2,b3,d1,d1abl,u1,d2,d3,v0 --probe-layers 0.5 --b2-max 1500
python tamperbench.py evaluate --n-boot 500
```

The second Qwen seed uses the same commands with `--seed 1`. The D3 control (report appendix,
Table C3) retrains on L0-L1 variants only:

```
python tamperbench.py detect --detectors b1,d2,d3 --probe-layers 0.5 --train-levels 0,1
python tamperbench.py evaluate --n-boot 200
```

It was run on a copy of the run directory that reuses the cached D2/D3 hidden states.

Swarm simulation: `python tamperbench.py swarm --n-episodes 200`.

Every stage is resumable: re-run the same command after an interruption, and
`python tamperbench.py status` shows progress. Other command-line options exist (for example other
probe layers or agent policies) but were not used for the reported results.

## Code version

`tamperbench.py` sha256 at run time: `a68697dcb421ddc20bc1be40f8999db0ddde54cf4dd1dd0ce4427a99ce48e950`.
The only change after the experiments is the module docstring (comments). The run manifests in
`results/` record the hashes of the code that produced each run.

## Pre-existing code and prior work

None. The sandbox, task families, tamper operators, detectors (including the D3 probe), evaluation code (tamperbench.py) and the driver (run_all.py) were written during the hackathon weekend; no code from earlier projects was reused. Standard libraries (numpy, scikit-learn, PyTorch, Hugging Face transformers) and the pretrained models Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct are used as published.

## Limitations

Small synthetic sandbox, one agent model for the main results (two generation seeds, same model and
code), a scripted swarm simulation, and detectors trained on the same operators they are tested on.
See the report for details, including 102 sanity problems that were not diagnosed.

## Dual-use note

The benchmark includes operators that rewrite logs to hide a benign synthetic action, and an L3
best-of-n attacker that picks among synthetic edits. They run only on synthetic logs in a toy
sandbox. See the report's dual-use appendix.

## License

Code: MIT. Data: CC BY 4.0. Data generated with Llama-3.1-8B-Instruct is
additionally subject to the Llama 3.1 Community License.

## Author

Vahideh Zolfaghari, vahidehzolfagharii@gmail.com
