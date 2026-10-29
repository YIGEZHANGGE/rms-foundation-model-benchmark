#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3: score S1-S4 on Xenium malignant cells and report S4 (IEG) contamination."""
import os
import csv
import anndata as ad
import numpy as np
from scipy import sparse

P = r"E:\rms_persister\data\xenium\processed_data\Spatial_hm_July2025_pub.h5ad"
SIG = r"E:\rms_persister\results\signatures"


def load_genes(fn, col="gene"):
    g = []
    with open(os.path.join(SIG, fn), encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            g.append(r[col])
    return g


sigs = {
    "S1_progenitor": load_genes("S1_progenitor.tsv"),
    "S2_treatment": load_genes("S2_treatment_enriched.tsv"),
    "S3_convergent": load_genes("S3_convergent.tsv"),
    "S4_ieg": load_genes("S4_ieg_stress.tsv"),
}

a = ad.read_h5ad(P, backed="r")
genes = list(a.var_names)
malig_mask = (a.obs["cell_class"] == "malignant").values

scores = {}
overlap = {}
for name, gs in sigs.items():
    present = [g for g in gs if g in genes]
    overlap[name] = (len(present), len(gs))
    idx = [genes.index(g) for g in present]
    if not idx:
        scores[name] = np.zeros(malig_mask.sum())
        continue
    Xn = a.X[malig_mask][:, idx]
    if sparse.issparse(Xn):
        Xn = Xn.toarray()
    Xn = np.asarray(Xn, dtype=np.float32)
    mu = Xn.mean(axis=0); sd = Xn.std(axis=0); sd[sd == 0] = 1.0
    Z = (Xn - mu) / sd
    scores[name] = Z.mean(axis=1)

print("signature gene overlap in Xenium panel:")
for k, (p, t) in overlap.items():
    print(f"  {k}: {p}/{t} genes present")

# correlation matrix
names = list(scores.keys())
print("\nSpearman correlation matrix (malignant cells):")
print(f"{'':>16}", *[f"{n:>16}" for n in names])
from scipy.stats import spearmanr
for n1 in names:
    row = []
    for n2 in names:
        r, _ = spearmanr(scores[n1], scores[n2])
        row.append(r)
    print(f"{n1:>16}", *[f"{v:>16.3f}" for v in row])

# S4 contamination conclusion
print("\nS4 (IEG) correlation with signatures:")
for n in ["S1_progenitor", "S2_treatment", "S3_convergent"]:
    r, p = spearmanr(scores["S4_ieg"], scores[n])
    flag = "REGESSION NEEDED (|r|>0.5)" if abs(r) > 0.5 else "OK"
    print(f"  S4 vs {n}: r={r:.3f}  -> {flag}")
a.file.close()
