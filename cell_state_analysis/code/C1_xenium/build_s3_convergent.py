#!/usr/bin/env python3
import os, csv
import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse
import pandas as pd

P = r"E:\rms_persister\data\xenium\processed_data\Spatial_hm_July2025_pub.h5ad"
OUT = r"E:\rms_persister\results\signatures"
os.makedirs(OUT, exist_ok=True)

NEURO = ["DCX","L1CAM","SYP","CHGA","NEFM","NEFH","STMN2","TUBB3","NCAM1","GAP43","NEFL","SNAP25","ELAVL4","RBFOX3","CHGB","SCG2","SYT1","MAP2","NSG1","NRN1"]

a = ad.read_h5ad(P, backed="r")
genes = list(a.var_names)
present = [g for g in NEURO if g in genes]
print("neuro markers present:", present)
neuro_idx = [genes.index(g) for g in present]
malig_mask = (a.obs["cell_class"] == "malignant").values

X = a.X
Xn = X[malig_mask][:, neuro_idx]
if sparse.issparse(Xn):
    Xn = Xn.toarray()
Xn = np.asarray(Xn, dtype=np.float32)
mu = Xn.mean(axis=0); sd = Xn.std(axis=0); sd[sd==0]=1.0
Z = (Xn - mu)/sd
neuro_score = Z.mean(axis=1)
print("malignant cells:", len(neuro_score))

thr_hi = np.quantile(neuro_score, 0.95)
thr_lo = np.quantile(neuro_score, 0.50)
hi = neuro_score >= thr_hi
lo = neuro_score <= thr_lo
print("high-neuro:", int(hi.sum()), "low-neuro:", int(lo.sum()))

rng = np.random.RandomState(42)
hi_idx = rng.choice(np.where(hi)[0], size=min(6000, int(hi.sum())), replace=False)
lo_idx = rng.choice(np.where(lo)[0], size=min(6000, int(lo.sum())), replace=False)
sel = np.sort(np.concatenate([hi_idx, lo_idx]))
group = np.where(np.isin(sel, hi_idx), "1", "0")

Xsel = X[malig_mask][sel, :]
if sparse.issparse(Xsel):
    Xsel = Xsel.toarray()
b = ad.AnnData(X=np.asarray(Xsel, dtype=np.float32),
               obs=pd.DataFrame({"group": group}, index=[str(i) for i in range(len(sel))]),
               var=pd.DataFrame(index=genes))
sc.tl.rank_genes_groups(b, groupby="group", reference="0", method="wilcoxon")
df = sc.get.rank_genes_groups_df(b, group="1")
df = df[df["pvals_adj"] < 0.05].sort_values("logfoldchanges", ascending=False)
print("DE genes (padj<0.05):", len(df))

sig = df.head(200)
with open(os.path.join(OUT, "S3_convergent.tsv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["gene","log2FC","padj"])
    for _, r in sig.iterrows():
        w.writerow([r["names"], round(r["logfoldchanges"],4), f"{r['pvals_adj']:.3g}"])
print("S3 saved:", len(sig), "genes (top:", ",".join(df["names"].head(12)), ")")
a.file.close()
