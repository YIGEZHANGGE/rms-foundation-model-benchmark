#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate A: score S1/S2/S3 on E-TABM-1202 and run Cox (survival ~ score + Age + IRS stage)."""
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
print("expr:", expr.shape, "| pheno:", pheno.shape)

# survival / covariates
time = pd.to_numeric(pheno["Factor.Value..survival_time."], errors="coerce")
event = (pheno["Factor.Value..Status."].astype(str).str.strip() == "dead from disease").astype(int)
age = pd.to_numeric(pheno["Factor.Value..Age."], errors="coerce")
stage = pd.to_numeric(pheno["Factor.Value..IRS_TNM_STAGE."], errors="coerce")
hist = pheno["Factor.Value..Histology2."].astype(str).str.strip()

# align samples
common = expr.columns.intersection(pheno.index)
expr = expr[common]
time = time.loc[common]; event = event.loc[common]; age = age.loc[common]
stage = stage.loc[common]; hist = hist.loc[common]
print("aligned samples:", len(common), "| events:", int(event.sum()))


def score_sig(genes):
    g = [x for x in genes if x in expr.index]
    if not g:
        return np.zeros(len(common))
    X = np.log2(expr.loc[g].T + 1)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0).replace(0, 1))
    return Z.mean(axis=1).values


sigs = {
    "S1_progenitor": score_sig(load_genes("S1_progenitor.tsv")),
    "S2_treatment": score_sig(load_genes("S2_treatment_enriched.tsv")),
    "S3_convergent": score_sig(load_genes("S3_convergent.tsv")),
    "S4_ieg": score_sig(load_genes("S4_ieg_stress.tsv")),
}


def cox(score, label):
    # model: survival ~ score + age + stage
    df = pd.DataFrame({"time": time.values, "event": event.values,
                       "score": score, "age": age.values, "stage": stage.values})
    df = df.dropna(subset=["time", "score", "age", "stage"])
    m = PHReg(endog=df["time"], exog=df[["score", "age", "stage"]],
              status=df["event"], ties="efron")
    r = m.fit()
    # PHReg returns params/pvalues as arrays aligned with exog columns (score first)
    coef = float(np.asarray(r.params)[0])
    hr = float(np.exp(coef))
    p = float(np.asarray(r.pvalues)[0])
    # log-rank-ish direction check: high vs low score (median split)
    print(f"{label}: n={len(df)} events={int(df['event'].sum())} "
          f"HR={hr:.3f} p={p:.4g} (cox score+age+stage)")
    return label, hr, p, len(df), int(df["event"].sum())


print("\n=== Cox: survival ~ score + age + IRS stage ===")
results = []
for label, s in sigs.items():
    results.append(cox(s, label))

print("\n=== Gate A decision ===")
gate = False
for label, hr, p, n, ev in results:
    if label in ("S1_progenitor", "S2_treatment", "S3_convergent"):
        # direction: high-risk state should have HR > 1 (consistent with literature)
        if p < 0.05 and hr > 1:
            print(f"  {label}: PASS (p={p:.4g}, HR={hr:.2f} > 1)")
            gate = True
        else:
            print(f"  {label}: not significant (p={p:.4g}, HR={hr:.2f})")
print("Gate A:", "PASS -> proceed to Phase C" if gate else "HOLD (no signature passed)")
