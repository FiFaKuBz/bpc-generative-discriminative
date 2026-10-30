"""Phase 1 final runs: pick the best grid configs and train every seed with --save.

* bPC     (alpha_gen 1e-4): best 2(1 - val_acc) + RMSE in results/sweep_bpc       -> 5 seeds
  (RMSE in the paper's units = ours / 2: the paper's RMSEs are on a [0, 1] pixel scale, see docs/REPLICATION.md)
* discPC  (alpha_gen 0)   : best val_acc in results/sweep_disc                   -> 5 seeds
* bPC at alpha_gen 1e-3 and 1e-2, bPC's other settings                           -> 3 seeds each
  (how much generation reshapes the inside as its weight grows)

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/run_final.py --workers 5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcn.bpc import BPCConfig
from scripts.train_bpc import RESULTS, name_of, run


def best(tag: str, key) -> tuple[BPCConfig, dict]:
    rows = [json.loads(p.read_text()) for p in (RESULTS / tag).glob("*.json")]
    rows = [r for r in rows if "val_acc" in r]
    r = min(rows, key=key)
    cfg = dict(r["config"])
    cfg["sizes"] = tuple(cfg["sizes"])
    return BPCConfig(**cfg), r


def _job(cfg: BPCConfig) -> dict:
    path = RESULTS / "final" / f"{name_of(cfg)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return run(cfg, "final", save=True, eval_every=5, verbose=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    b, rb = best("sweep_bpc", lambda r: 2 * (1 - r["val_acc"]) + r["val_rmse"] / 2)
    dsc, rd = best("sweep_disc", lambda r: -r["val_acc"])
    sel = {"bPC": {"name": rb["name"], "val_acc": rb["val_acc"], "val_rmse": rb["val_rmse"]},
           "discPC": {"name": rd["name"], "val_acc": rd["val_acc"], "val_rmse": rd["val_rmse"]}}
    (RESULTS / "final").mkdir(parents=True, exist_ok=True)
    (RESULTS / "final" / "selected.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1), flush=True)
    cfgs = [replace(b, seed=s) for s in range(5)] + [replace(dsc, seed=s) for s in range(5)]
    cfgs += [replace(b, a_gen=ag, seed=s) for ag in (1e-3, 1e-2) for s in range(3)]
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(_job, c) for c in cfgs]):
            r = f.result()
            print(f"{r['name']}  test_acc={r.get('test_acc')}  test_rmse={r.get('test_rmse')}", flush=True)


if __name__ == "__main__":
    main()
