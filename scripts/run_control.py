"""Control: discPC with bPC's weight decay (1e-4), so bPC vs discPC differ only in alpha_gen.

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/run_control.py --workers 5
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcn.bpc import BPCConfig
from scripts.train_bpc import RESULTS, name_of, run

def _job(cfg):
    path = RESULTS / "final" / f"{name_of(cfg)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return run(cfg, "final", save=True, eval_every=5, verbose=False)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=5); a = p.parse_args()
    sel = json.loads((RESULTS / "final" / "selected.json").read_text())
    cfg = dict(json.loads((RESULTS / "final" / f"{sel['bPC']['name']}.json").read_text())["config"])
    cfg["sizes"] = tuple(cfg["sizes"])
    b = BPCConfig(**cfg)
    cfgs = [replace(b, a_gen=0.0, seed=s) for s in range(5)]   # same as bPC except alpha_gen
    print("control:", name_of(cfgs[0]), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(_job, c) for c in cfgs]):
            r = f.result(); print(f"{r['name']}  test_acc={r.get('test_acc')}", flush=True)

if __name__ == "__main__":
    main()
