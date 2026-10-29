#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate A mandatory checks: (B) S1/S2 collinearity; (A) proliferation confounding."""
import os
import csv
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

D = r"E:\rms_persister\data\target"
SIG = r"E:\rms_persister\results\signatures"


def load_genes(fn):
    g = []
    with open(os.path.join(SIG, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            g.append(r["gene"])
    return g


expr = pd.read_csv(os.path.join(D, "E-TABM-1202.gene_expr.tsv"), sep="\t", index_col=0)
pheno = pd.read_csv(os.path.join(D, "E-TABM-1202.pheno.tsv"), sep="\t", index_col=0)
common = expr.columns.intersection(pheno.index)
expr = expr[common]
time = pd.to_numeric(pheno.loc[common, "Factor.Value..survival_time."], errors="coerce")
event = (pheno.loc[common, "Factor.Value..Status."].astype(str).str.strip() == "dead from disease").astype(int)
age = pd.to_numeric(pheno.loc[common, "Factor.Value..Age."], errors="coerce")
stage = pd.to_numeric(pheno.loc[common, "Factor.Value..IRS_TNM_STAGE."], errors="coerce")


def score(genes):
    g = [x for x in genes if x in expr.index]
    X = np.log2(expr.loc[g].T + 1)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0).replace(0, 1))
    return Z.mean(axis=1).values, len(g)


# ---- Check B: S1/S2 collinearity ----
s1_genes = load_genes("S1_progenitor.tsv")
s2_genes = load_genes("S2_treatment_enriched.tsv")
overlap = len(set(s1_genes) & set(s2_genes))
union = len(set(s1_genes) | set(s2_genes))
s1_score, _ = score(s1_genes)
s2_score, _ = score(s2_genes)
rho, p = stats.spearmanr(s1_score, s2_score)
print(f"[B] S1/S2: gene overlap {overlap}/{union} (Jaccard {overlap/union:.3f}), "
      f"score Spearman rho={rho:.4f} (p={p:.3g})")
merge = rho > 0.9
print(f"[B] merge into single hypothesis: {merge}  (BH denominator -> 2)")

# ---- Check A: proliferation confounding ----
PROLIF = ["MKI67", "TOP2A", "E2F1", "E2F2", "E2F7", "E2F8", "PCNA", "CDK1",
          "CDK2", "CCNB1", "CCNB2", "CCNA2", "MCM2", "MCM3", "MCM4", "MCM5",
          "MCM6", "MCM7", "AURKA", "AURKB", "BIRC5", "PLK1", "CDC20", "CENPA",
          "KIF11", "KIF23", "NUSAP1", "UBE2C", "TPX2", "CKS2"]
prolif_score, n_prolif = score(PROLIF)
print(f"[A] proliferation signature: {n_prolif} genes")


def cox(score, label, extra=None):
    cols = {"time": time.values, "event": event.values, "score": score,
            "age": age.values, "stage": stage.values}
    if extra is not None:
        cols["extra"] = extra
    df = pd.DataFrame(cols).dropna()
    exog_cols = ["score", "age", "stage"] + (["extra"] if extra is not None else [])
    m = PHReg(endog=df["time"], exog=df[exog_cols], status=df["event"], ties="efron")
    r = m.fit()
    coef = float(np.asarray(r.params)[0])
    hr = float(np.exp(coef))
    p = float(np.asarray(r.pvalues)[0])
    return hr, p


hr_base, p_base = cox(s1_score, "S1 base")
hr_prolif, p_prolif = cox(s1_score, "S1 + proliferation", extra=prolif_score)
print(f"[A] S1 HR base = {hr_base:.3f} (p={p_base:.4g})")
print(f"[A] S1 HR + proliferation covariate = {hr_prolif:.3f} (p={p_prolif:.4g})")
delta = hr_prolif - hr_base
concl = "attenuated -> proliferation proxy" if abs(delta) > 0.10 else "held -> state-specific"
print(f"[A] HR change = {delta:+.3f} -> {concl}")
