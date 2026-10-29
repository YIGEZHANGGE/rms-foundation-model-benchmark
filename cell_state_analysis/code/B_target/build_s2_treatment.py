#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build S2 (treatment-enriched signature) from Danielli bulk GSE240308 (mouse PDX, VCR+IRN vs vehicle)."""
import os
import csv
import gzip
import requests
import numpy as np
import pandas as pd
from scipy import stats

TPM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE240nnn/GSE240308/suppl/GSE240308_DYER-223923-STRANDED_RSEM_TPM.txt.gz"
OUT = r"E:\rms_persister\results\signatures"
os.makedirs(OUT, exist_ok=True)

# sample -> (mouse, treatment, day) from series matrix
raw = gzip.decompress(requests.get(
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE240nnn/GSE240308/matrix/GSE240308_series_matrix.txt.gz",
    timeout=30).content).decode("utf-8", errors="replace")
lines = raw.split("\n")
titles = [l for l in lines if l.startswith("!Sample_title")][0].split("\t")[1:]
chars = [l for l in lines if l.startswith("!Sample_characteristics_ch1")]

meta = {}
for i, t in enumerate(titles):
    t = t.strip('"')
    mouse = treatment = day = ""
    for c in chars:
        vals = c.split("\t")[1:]
        if i < len(vals):
            v = vals[i].strip('"')
            if "mouse number" in c:
                mouse = v.split(":")[-1].strip()
            if c.strip().startswith("!Sample_characteristics_ch1") and "treatment" in c and "time" not in c:
                treatment = v.split(":")[-1].strip()
            if "time point" in c:
                day = v.split(":")[-1].strip()
    meta[t] = (mouse, treatment, day)

# download TPM
tpm_raw = gzip.decompress(requests.get(TPM_URL, timeout=60).content).decode("utf-8", errors="replace")
df = pd.read_csv(pd.io.common.StringIO(tpm_raw), sep="\t", index_col=0)
print("TPM shape:", df.shape, "| columns sample-like:", [c for c in df.columns][:5])

# map columns (sample titles) to meta
# GEO compiled TPM columns are sample titles like "SHAGGY_6260 day 0"
naive_cols = [c for c in df.columns if "day 0" in c]
therapy_cols = [c for c in df.columns if ("day 7" in c or "day 14" in c) and "6260" in c or "6259" in c]
therapy_cols = [c for c in df.columns if (("day 7" in c) or ("day 14" in c))]
print("naive cols:", naive_cols)
print("therapy cols:", therapy_cols)

# compute log2FC + t-test (naive vs on-therapy)
naive = df[naive_cols].apply(np.log2, axis=0).apply(lambda x: x.replace([np.inf, -np.inf], np.nan))
therapy = df[therapy_cols].apply(np.log2, axis=0).apply(lambda x: x.replace([np.inf, -np.inf], np.nan))

res = []
for g in df.index:
    a = naive.loc[g].dropna().values
    b = therapy.loc[g].dropna().values
    if len(a) < 2 or len(b) < 2:
        continue
    fc = np.mean(b) - np.mean(a)
    t, p = stats.ttest_ind(a, b)
    res.append((g, fc, p))

r = pd.DataFrame(res, columns=["gene", "log2FC", "pval"])
r["padj"] = r["pval"] * len(r)  # Bonferroni-ish
r = r[(r["padj"] < 0.05) & (r["log2FC"].notna())].sort_values("log2FC", ascending=False)
print("S2 DE genes (padj<0.05):", len(r))

sig = r.head(250)
with open(os.path.join(OUT, "S2_treatment_enriched.tsv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["gene", "log2FC", "padj"])
    for _, row in sig.iterrows():
        w.writerow([row["gene"], round(row["log2FC"], 4), f"{row['padj']:.3g}"])
print("S2 saved:", len(sig), "genes (top:", ", ".join(r["gene"].head(12)), ")")
