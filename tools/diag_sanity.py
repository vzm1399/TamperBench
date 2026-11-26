#!/usr/bin/env python3
"""Diagnose why untampered logs fail their own consistency checks.
Usage (in the folder with tamperbench.py):  python diag_sanity.py runs/qwen2.5-7b_seed0/episodes.jsonl
Prints a SMALL summary you can paste back. Writes nothing except diag_sanity_out.json."""
import sys, re, json, collections
import tamperbench as tb

path = sys.argv[1]
eps = tb.read_jsonl(path)
def valid(e):  # same filter as cmd_tamper
    return (e["success"] and (e["condition"] == "clean" or e["violation_done"]) and len(e["steps"]) >= 3
            and (e["condition"] == "violation" or not e["violation_steps"]))
eps = [e for e in eps if valid(e)]
PYWRITE = re.compile(r"open\s*\([^)]*,\s*['\"][wax+]|\.write_text|\.write_bytes|\.write\s*\(|shutil|os\.rename|os\.replace")
rows, tasks = [], {}
for e in eps:
    key = (e["family"], e["seed"]); t = tasks.setdefault(key, tb.make_task(*key))
    fl, g = tb.d1_scores(t, e["steps"], e["host"], e["final_state_hash"])
    ul, ug = tb.u1_scores(t, e["steps"], e["final_state_hash"])
    py = [s for s in e["steps"] if s["tool"] == "run_python"]
    pyw = [s for s in py if PYWRITE.search(s["args"].get("code", "") if isinstance(s["args"], dict) else "")]
    rows.append(dict(id=e["episode_id"], family=e["family"], cond=e["condition"], pure=e.get("llm_pure"),
                     d1=bool(sum(fl) + g), u1=bool(sum(ul) + ug), d1_hash=bool(g >= 2), d1_len=bool(int(g) % 2 == 1),
                     has_py=bool(py), py_write=bool(pyw)))
bad = [r for r in rows if r["d1"] or r["u1"]]
print(f"valid episodes: {len(rows)} | failing own checks: {len(bad)}  (tamper log said 102)")
def tab(title, keyf):
    c = collections.Counter(keyf(r) for r in bad); a = collections.Counter(keyf(r) for r in rows)
    print(f"\n{title}"); [print(f"  {k}: {c[k]}/{a[k]}") for k in sorted(a, key=str)]
tab("by family (failing/all)", lambda r: r["family"])
tab("by condition", lambda r: r["cond"])
tab("by who flags", lambda r: ("D1 only" if r["d1"] and not r["u1"] else "U1 only" if r["u1"] and not r["d1"] else "both" if r["d1"] else "none"))
tab("uses run_python that WRITES a file", lambda r: r["py_write"])
print("\nfailing episodes WITHOUT a run_python write (need another explanation):")
rest = [r for r in bad if not r["py_write"]]
print(f"  {len(rest)}"); [print("   ", r["id"], "D1" if r["d1"] else "", "U1" if r["u1"] else "") for r in rest[:15]]
json.dump(bad, open("diag_sanity_out.json", "w"), indent=1)
