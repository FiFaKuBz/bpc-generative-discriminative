"""Rebuild test (instance id among 100 same-digit probes) for the control discPC (wd 1e-4)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcn.bpc_data import mnist_split
from scripts.inside import load_net, ladder, mean_sem, FINAL
torch.set_num_threads(2)
d = mnist_split(); xp, yp = d.probe; means = d.class_means("test")
out = {}
for tag, pat in (("discPC_wd1e-4", "agen0_*wd0.0001_*e25_s*.json"),):
    rows = []
    for p in sorted(FINAL.glob(pat)):
        r = json.loads(p.read_text())
        if "test_acc" not in r: continue
        rows.append(ladder(load_net(r), xp, yp, means)[0])
    out[tag] = {l: {m: mean_sem([row[l][m] for row in rows]) for m in rows[0][l]} for l in rows[0]}
print(json.dumps(out, indent=1))
(FINAL / "rebuild_control.json").write_text(json.dumps(out, indent=1))
