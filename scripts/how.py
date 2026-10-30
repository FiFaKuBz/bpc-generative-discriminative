"""Phase 2b: HOW does learning to generate change the layers? (CKA only says where / how much.)

Works on the saved probe states (1,000 test images, 100 per digit, after settling with the
image clamped, T_eval = 100), final epoch, every condition in results/final/.

  G  geometry per layer (mean ± s.e.m. over seeds)
       knn        10-NN digit accuracy, 5-fold          (are same digits neighbours?)
       silhouette silhouette by digit, cosine           (how tight and separate are the clusters?)
       within     within-digit variance / total variance (how much of the spread is handwriting, not digit?)
       pr         participation ratio of the covariance (how many directions the layer really uses)
       style_rsa  Spearman between same-digit pair distances in the layer and in pixels
                  (does the layer keep *which* 3s look alike in the image?)
  H  stability: first epoch after which validation accuracy stays within 0.3 points of epoch 25
  I  CKA control: bPC vs discPC trained with bPC's weight decay (only alpha_gen differs)
  figures: t-SNE grid, similarity-matrix (RSM) intuition, training curves

    PYTHONPATH=. python3 scripts/how.py
"""
from __future__ import annotations

import itertools, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcn.bpc_data import mnist_split
from pcn.metrics import linear_cka
from pcn.stats import compare_to_floor

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "results" / "final"
LAYERS = ("L1", "L2", "L3")


def cond(cfg) -> str | None:
    a, wd = cfg["a_gen"], cfg["wd"]
    if a == 0:
        return "discPC" if wd == 1e-3 else "discPC_wd1e-4"
    return {1e-4: "bPC", 1e-3: "bPC_1e-3", 1e-2: "bPC_1e-2"}.get(a)


def load():
    runs = {}
    for p in sorted(FINAL.glob("agen*.json")):
        r = json.loads(p.read_text())
        npz = FINAL / f"{r['name']}__probe.npz"
        if "test_acc" not in r or not npz.exists():
            continue
        z = np.load(npz)
        ep = r["config"]["epochs"]
        r["S"] = {l: z[f"e{ep}_{l}"].astype(np.float64) for l in LAYERS}
        r["snaps"] = {k: z[k] for k in z.files}
        runs.setdefault(cond(r["config"]), []).append(r)
    bp = json.loads((FINAL / "bp.json").read_text())
    runs["BP"] = [{"S": {l: np.load(FINAL / f"bp_s{s}__probe.npz")[l].astype(np.float64) for l in LAYERS}}
                  for s in range(len(bp["seeds"]))]
    return runs


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0]


def pdist_same(X, y):
    out = []
    for c in range(10):
        Z = X[y == c]
        D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        out.append(D[np.triu_indices(len(Z), 1)])
    return np.concatenate(out)


def geometry(X, y, pix_d):
    Xc = X - X.mean(0)
    knn = cross_val_score(KNeighborsClassifier(10), X, y, cv=5).mean()
    sil = silhouette_score(X, y, metric="cosine")
    tot = (Xc ** 2).sum()
    within = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum() for c in range(10)) / tot
    ev = np.linalg.eigvalsh(np.cov(Xc.T))
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    rsa = spearmanr(pdist_same(X, y), pix_d)[0]
    return dict(knn=knn, silhouette=sil, within=within, pr=pr, style_rsa=rsa)


def stable_epoch(r, tol=0.3):
    h = [(e["epoch"], 100 * e["val_acc"]) for e in r["history"] if "val_acc" in e]
    final = h[-1][1]
    for i, (ep, _) in enumerate(h):
        if all(abs(a - final) <= tol for _, a in h[i:]):
            return ep
    return h[-1][0]


def cka_block(A, B):
    out = {}
    for l in LAYERS:
        t = lambda r: torch.from_numpy(r["S"][l])
        floor = [1 - linear_cka(t(a), t(b)) for a, b in itertools.combinations(A, 2)]
        floor += [1 - linear_cka(t(a), t(b)) for a, b in itertools.combinations(B, 2)]
        cross = [1 - linear_cka(t(a), t(b)) for a in A for b in B]
        e = compare_to_floor(np.array(cross), np.array(floor), n_perm=5000).to_dict()
        out[l] = dict(floor=float(np.mean(floor)), cross=float(np.mean(cross)),
                      ratio=float(np.mean(cross) / np.mean(floor)), d=e["cohens_d"], auc=e["auc"], p=e["perm_p"],
                      n_floor=len(floor), n_cross=len(cross), no_overlap=bool(min(cross) > max(floor)))
    return out


def main():
    d = mnist_split()
    xp, yp = d.probe
    X0, y = xp.numpy().astype(np.float64), yp.numpy()
    pix_d = pdist_same(X0, y)
    runs = load()
    order = [c for c in ("discPC", "discPC_wd1e-4", "bPC", "bPC_1e-3", "bPC_1e-2", "BP") if c in runs]
    res = {"n": {c: len(runs[c]) for c in order}, "G": {}, "H": {}, "I": {}}
    res["G"]["pixels"] = {k: float(v) for k, v in geometry(X0, y, pix_d).items()}
    for c in order:
        rows = [{l: geometry(r["S"][l], y, pix_d) for l in LAYERS} for r in runs[c]]
        res["G"][c] = {l: {m: ms([row[l][m] for row in rows]) for m in rows[0][l]} for l in LAYERS}
        print(c, json.dumps({l: {m: round(v[0], 3) for m, v in res["G"][c][l].items()} for l in LAYERS}), flush=True)
    for c in order:
        if c == "BP":
            continue
        res["H"][c] = {"stable_epoch": [stable_epoch(r) for r in runs[c]],
                       "acc_e25": ms([100 * r["test_acc"] for r in runs[c]]),
                       "rmse_e25_paper_units": ms([r["test_rmse"] / 2 for r in runs[c]])}
    print(json.dumps(res["H"], indent=1), flush=True)
    if "discPC_wd1e-4" in runs:
        res["I"]["bPC_vs_discPC_wd1e-4"] = cka_block(runs["bPC"], runs["discPC_wd1e-4"])
        res["I"]["bPC_vs_discPC"] = cka_block(runs["bPC"], runs["discPC"])
        res["I"]["discPC_wd1e-4_vs_discPC"] = cka_block(runs["discPC_wd1e-4"], runs["discPC"])
        for c in ("bPC_1e-3", "bPC_1e-2"):
            res["I"][f"{c}_vs_discPC_wd1e-4"] = cka_block(runs[c], runs["discPC_wd1e-4"])
        print(json.dumps(res["I"], indent=1), flush=True)
    # t-SNE embeddings (seed 0), saved for the figure script
    emb = {"y": y}
    emb["pixels"] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X0)
    for c in ("discPC", "discPC_wd1e-4", "bPC"):
        if c in runs:
            for l in LAYERS:
                emb[f"{c}_{l}"] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(runs[c][0]["S"][l])
    np.savez(FINAL / "how_tsne.npz", **emb)
    (FINAL / "how.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
