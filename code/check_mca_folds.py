# -*- coding: utf-8 -*-
"""Verify whether CONCH/MultiClusterAttention folds 3 and 4 produce different per-patient predictions."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from architecture_ablation_v3 import train_one, ENCODERS, COHORT_PATH, OUT_DIR

cohort = json.load(open(COHORT_PATH, encoding="utf-8"))
labels = {str(k): int(v) for k, v in cohort["labels"].items()}
folds = cohort["folds"]
feat_dir, dim = ENCODERS["conch"]
feat_files = {os.path.basename(f).replace(".h5", ""): f for f in glob.glob(os.path.join(feat_dir, "*.h5"))}

out = {}
for fi in (1, 2, 3, 4, 5):
    fd = folds[f"fold_{fi}"]
    tr = [p for p in fd["train_pids"] if p in feat_files]
    va = [p for p in fd["val_pids"] if p in feat_files]
    r = train_one("MultiClusterAttention", dim, feat_files, tr, va, labels, fi)
    out[f"fold_{fi}"] = {"pids": r["pids"], "preds": [int(x) for x in r["preds"]], "trues": [int(x) for x in r["trues"]], "acc": r["acc"]}
    print(f"fold {fi}: acc={r['acc']:.4f} n={len(r['pids'])}", flush=True)

json.dump(out, open(os.path.join(OUT_DIR, "mca_conch_fold_preds.json"), "w"), indent=1)
# fold 3 vs 4 逐例比较
a = dict(zip(out["fold_3"]["pids"], out["fold_3"]["preds"]))
b = dict(zip(out["fold_4"]["pids"], out["fold_4"]["preds"]))
common = set(a) & set(b)
diff = [p for p in common if a[p] != b[p]]
print("overlap patients:", len(common), "| differing predictions:", len(diff))
print("fold3 pids == fold4 pids ?", set(a) == set(b))
