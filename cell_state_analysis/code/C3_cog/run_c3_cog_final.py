#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 COG primary analysis: H3a (S1 protective) / H3b (S3 risk), pre-registered.

Model: EFS ~ score + age_at_diagnosis + stage  (score z-scored per cohort, HR per 1-SD).
Encoding (per spec): efs_event/os_event NaN->0; time in DAYS -> /365.25 years.
os_days_s3 NaN (139) -> excluded from OS set. cog_risk_group complete-case sensitivity.
"""
import json
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

CLIN = r"E:\rms_persister\data\cog_clinical\COG_RMS_patient_table.csv"
PRED = r"E:\rms_persister\results\cog_pred\cog_slide_scores.csv"
COHORT = r"C:\Users\onekey\Documents\Codex\2026-08-02\read-project-context-md-in-this\outputs\data_snapshot\data\cohort_and_folds_public.json"


def cox(d, tcol, ecol, scol):
    d = d[[tcol, ecol, scol, "age", "stage"]].dropna()
    d = d[d[tcol] > 0]
    if d[ecol].nunique() < 2 or len(d) < 30:
        return None
    m = PHReg(endog=d[tcol], exog=d[[scol, "age", "stage"]], status=d[ecol], ties="efron")
    r = m.fit()
    hr = float(np.exp(float(np.asarray(r.params)[0])))
    p = float(np.asarray(r.pvalues)[0])
    # Harrell C-index
    risk = d[scol].values; t = d[tcol].values; e = d[ecol].values
    conc = denom = 0.0
    for i in range(len(d)):
        if e[i] != 1:
            continue
        for j in range(len(d)):
            if t[j] > t[i]:
                denom += 1
                conc += 1.0 if risk[i] > risk[j] else (0.5 if risk[i] == risk[j] else 0.0)
    cidx = conc / denom if denom else float("nan")
    return {"n": len(d), "events": int(d[ecol].sum()), "HR": hr, "p": p, "cindex": cidx}


def main():
    clin = pd.read_csv(CLIN)
    pred = pd.read_csv(PRED).rename(columns={"slide": "pid"})
    pred["pid"] = pred["pid"].astype(str)
    cohort_ids = set(json.load(open(COHORT, encoding="utf-8"))["labels"].keys())

    clin = clin[clin["patient_id"].isin(cohort_ids)]
    clin = clin.rename(columns={"patient_id": "pid"})
    clin["pid"] = clin["pid"].astype(str)
    df = pred[["pid", "S1", "S3", "S4"]].merge(clin, on="pid", how="inner")
    print(f"merged: {len(df)} patients", flush=True)

    # encoding
    df["efs_time"] = pd.to_numeric(df["efs_days"], errors="coerce") / 365.25
    df["efs_event"] = df["efs_event"].fillna(0).astype(int)
    df["os_time"] = pd.to_numeric(df["os_days_s3"], errors="coerce") / 365.25
    df["os_event"] = df["os_event"].fillna(0).astype(int)
    df["age"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
    df["risk"] = df["cog_risk_group"].astype(str).str.lower()
    df["risk_high"] = df["risk"].str.contains("high").astype(int)

    # verification
    efs_n = df["efs_time"].notna().sum()
    efs_ev = df.loc[df["efs_time"].notna(), "efs_event"].sum()
    os_n = df["os_time"].notna().sum()
    os_ev = df.loc[df["os_time"].notna(), "os_event"].sum()
    print(f"VERIFY EFS: n={efs_n} events={efs_ev}  (expect ~389/75)", flush=True)
    print(f"VERIFY OS : n={os_n} events={os_ev}  (expect ~258/50)", flush=True)
    if not (abs(efs_n - 389) <= 1 and efs_ev == 75 and os_n == 258 and os_ev == 50):
        print("!! MISMATCH vs manuscript — STOPPING", flush=True)
        return

    # z-score scores (per cohort)
    for s in ["S1", "S3", "S4"]:
        df[s] = (df[s] - df[s].mean()) / df[s].std()

    print("\n=== PRIMARY (COG, pre-registered) — EFS ~ score + age + stage ===", flush=True)
    r_s1 = cox(df, "efs_time", "efs_event", "S1")
    r_s3 = cox(df, "efs_time", "efs_event", "S3")
    r_s4 = cox(df, "efs_time", "efs_event", "S4")
    for name, r in [("S1", r_s1), ("S3", r_s3), ("S4", r_s4)]:
        if r:
            print(f"  {name}: n={r['n']} events={r['events']} HR={r['HR']:.3f} p={r['p']:.4g} C={r['cindex']:.3f}", flush=True)
        else:
            print(f"  {name}: insufficient", flush=True)

    # OS (secondary)
    print("\n=== OS (secondary endpoint) ===", flush=True)
    for name, r in [("S1", cox(df, "os_time", "os_event", "S1")),
                    ("S3", cox(df, "os_time", "os_event", "S3"))]:
        if r:
            print(f"  {name}: n={r['n']} events={r['events']} HR={r['HR']:.3f} p={r['p']:.4g}", flush=True)

    # sensitivity: + cog_risk_group (complete-case)
    print("\n=== sensitivity: EFS ~ score + age + stage + risk (complete-case) ===", flush=True)
    sub = df.dropna(subset=["risk"])
    sub = sub[sub["risk"].isin(["low", "intermediate", "high"])]
    for name in ["S1", "S3"]:
        d = sub[[name, "efs_time", "efs_event", "age", "stage", "risk_high"]].dropna()
        d = d[d["efs_time"] > 0]
        m = PHReg(endog=d["efs_time"], exog=d[[name, "age", "stage", "risk_high"]], status=d["efs_event"], ties="efron")
        r = m.fit()
        hr = float(np.exp(float(np.asarray(r.params)[0])))
        p = float(np.asarray(r.pvalues)[0])
        print(f"  {name}: n={len(d)} HR={hr:.3f} p={p:.4g}", flush=True)

    # H3a/H3b decision (BH k=2)
    print("\n=== H3a/H3b decision (BH, k=2) ===", flush=True)
    ps = {"H3a_S1": r_s1["p"] if r_s1 else 1.0, "H3b_S3": r_s3["p"] if r_s3 else 1.0}
    ranked = sorted(ps.items(), key=lambda x: x[1])
    bh = {k: p * 2 / (i + 1) for i, (k, p) in enumerate(ranked)}
    for k in ["H3a_S1", "H3b_S3"]:
        r = r_s1 if k == "H3a_S1" else r_s3
        hr = r["HR"] if r else float("nan")
        print(f"  {k}: raw p={ps[k]:.4g} BH p={bh[k]:.4g} HR={hr:.3f}", flush=True)
    h3a = r_s1 and bh["H3a_S1"] < 0.05 and r_s1["HR"] < 1
    h3b = r_s3 and bh["H3b_S3"] < 0.05 and r_s3["HR"] > 1
    print(f"  H3a (S1 protective HR<1): {'SUPPORTED' if h3a else 'not supported'}", flush=True)
    print(f"  H3b (S3 risk HR>1): {'SUPPORTED' if h3b else 'not supported'}", flush=True)


if __name__ == "__main__":
    main()
