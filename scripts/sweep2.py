"""Second search stage for bPC: activation (Table 5 options) and inference momentum,
at the learning rates that did best in stage 1. Same tag, so run_final.py ranks both stages.

    PYTHONPATH=. ../PCN-Version2/.venv/bin/python scripts/sweep2.py --workers 2
"""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcn.bpc import BPCConfig
from scripts.sweep import _job

CFGS = ([BPCConfig(act=a, lr_p=lp, wd=wd) for a in ("gelu", "tanh") for lp in (3e-4, 1e-4) for wd in (0.0, 1e-3)]
        + [BPCConfig(mom_x=m, lr_p=1e-4, wd=0.0) for m in (0.5, 0.9)])

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=2); a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(_job, c, "sweep_bpc") for c in CFGS]):
            r = f.result(); print(f"{r['name']}  val_acc={r.get('val_acc')}  val_rmse={r.get('val_rmse')}", flush=True)
