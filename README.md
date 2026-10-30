# One Network That Can Be Both Generative and Discriminative: Predictive Coding

Code and results for the blog post of the same name (Medium). It rebuilds
**bidirectional predictive coding (bPC)** from Oliviers, Tang & Bogacz (ICLR 2026) on
MNIST, then asks what switching on the drawing part (α_gen) changes inside the network.

**Main finding:** bPC and a read-only twin (discPC, α_gen = 0) differ most in the
**first** hidden layer, L1 (2.1× the null baseline; 2.0× at matched accuracy). bPC's L1
stays closer to the raw pixels (CKA 0.91 vs 0.78) and keeps details such as slant.

## Layout

```
pcn/
  bpc.py          bPC / discPC model: same layers, two weight sets (up, dn)
  data.py         downloads MNIST (into data/)
  bpc_data.py     MNIST train / validation / test split
  metrics.py      linear CKA
  stats.py        comparison against the null baseline (same-recipe pairs)
scripts/
  train_bpc.py    train one bPC / discPC run
  sweep.py, sweep2.py   grid search on validation data
  run_final.py    final runs for every α_gen (0, 1e-4, 1e-3, 1e-2)
  run_control.py  discPC with weight decay matched to bPC
  train_bp.py     backprop baseline
  inside.py       per-layer CKA vs null baseline (+ matched-accuracy check)
  how.py          CKA vs pixels / answers, t-SNE, silhouette
  rebuild_control.py, data_stats.py
results/final/    every run's JSON plus the summary files used in the post
figures/          scripts for the figures in the post, and their PNGs in figures/out/
```

## Run

```bash
pip install -r requirements.txt
python -m scripts.run_final      # trains the networks (MNIST downloads automatically)
python -m scripts.inside         # per-layer CKA results -> results/final/inside.json
python -m scripts.how            # how the layers change -> results/final/how.json
python figures/agen_s14.py       # example figure
```

## Reference

Oliviers, Tang & Bogacz (ICLR 2026), *Bidirectional predictive coding*.
