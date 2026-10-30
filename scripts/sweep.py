"""Grid search for the reproduction (phase 1).

The paper reports its search space (Table 5, SM D) but not the values it picked
for §4.1, so we search a grid inside that space on the validation split and
pick by the paper's own objectives: 2(1 - acc) + RMSE for bPC (joint), accuracy
alone for discPC (A.4).

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/sweep.py --grid bpc --workers 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcn.bpc import BPCConfig
from scripts.train_bpc import RESULTS, name_of, run

# SM D grid (lr_x, lr_p, weight decay) crossed with the Table 5 options we can afford.
LR_X = (0.01, 0.003, 0.001)
LR_P = (3e-4, 1e-4)          # 1e-3 dropped after stage 1 (RMSE 0.14-0.15)
WD = (0.0, 1e-4, 1e-3)


def grid(kind: str, form: str, act: str, a_gen: float) -> list[BPCConfig]:
    a_gens = (a_gen,) if kind == "bpc" else (0.0,)
    return [BPCConfig(a_gen=a, act=act, form=form, lr_x=lx, lr_p=lp, wd=wd)
            for a, lx, lp, wd in itertools.product(a_gens, LR_X, LR_P, WD)]


def _job(cfg: BPCConfig, tag: str) -> dict:
    path = RESULTS / tag / f"{name_of(cfg)}.json"
    if path.exists():                                   # resumable
        return json.loads(path.read_text())
    return run(cfg, tag, save=False, eval_every=25, verbose=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=("bpc", "disc"), required=True)
    p.add_argument("--form", default="fW")
    p.add_argument("--act", default="leaky_relu")
    p.add_argument("--a_gen", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    tag = f"sweep_{a.grid}"
    cfgs = grid(a.grid, a.form, a.act, a.a_gen)
    print(f"{len(cfgs)} configs -> results/{tag}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_job, c, tag): c for c in cfgs}
        for f in as_completed(futs):
            r = f.result()
            print(f"{r['name']}  val_acc={r.get('val_acc')}  val_rmse={r.get('val_rmse')}", flush=True)


if __name__ == "__main__":
    main()
