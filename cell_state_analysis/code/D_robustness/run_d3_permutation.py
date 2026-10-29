#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D3 — permutation null for S1/S3 vs random signatures in COG.

Samples 1,000 size-matched random signatures from the DeepSpot-M predicted
gene space, scores each slide, runs the same Cox model (EFS and OS), and
reports where the observed S1/S3 HR falls in the null distribution.
"""
import json
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

OUT = r"E:\rms_persister\results\D_robustness"
CLIN = r"E:\rms_persister\data\cog_clinical\COG_RMS_patient_table.csv"
COHORT = r"C:\Users\onekey\Documents\Codex\2026-08-02\read-project-context-md-in-this\outputs\data_snapshot\data\cohort_and_folds_public.json"
SIG = r"E:\rms_persister\results\signatures"
NPERM = 1000
SEED = 42


def load_genes(fn):
    g = []
    with open(fn, encoding="utf-8") as f:
        import csv
        for r in csv.DictReader(f, delimiter="\t"):
            g.append(r["gene"])
    return g


def cox_hr(d, tcol, ecol, scol):
    dd = d[[tcol, ecol, scol, "age", "stage"]].dropna()
    dd = dd[dd[tcol] > 0]
    if dd[ecol].nunique() < 2 or len(dd) < 30:
        return None
    m = PHReg(endog=dd[tcol], exog=dd[[scol, "age", "stage"]], status=dd[ecol], ties="efron")
    r = m.fit()
    return float(np.exp(float(np.asarray(r.params)[0]))), float(np.asarray(r.pvalues)[0])


def main():
    mat = np.load(f"{OUT}/D3_slide_gene_mean.npy")
    slides = open(f"{OUT}/D3_slides.txt").read().splitlines()
    genes = open(f"{OUT}/D3_genes.txt").read().splitlines()
    gene2i = {g: i for i, g in enumerate(genes)}
    print(f"matrix {mat.shape}; slides {len(slides)}; genes {len(genes)}")

    cohort_ids = set(json.load(open(COHORT, encoding="utf-8"))["labels"].keys())
    clin = pd.read_csv(CLIN).rename(columns={"patient_id": "pid"})
    clin["pid"] = clin["pid"].astype(str)
    clin = clin[clin["pid"].isin(cohort_ids)]
    clin["efs_time"] = pd.to_numeric(clin["efs_days"], errors="coerce") / 365.25
    clin["efs_event"] = clin["efs_event"].fillna(0).astype(int)
    clin["os_time"] = pd.to_numeric(clin["os_days_s3"], errors="coerce") / 365.25
    clin["os_event"] = clin["os_event"].fillna(0).astype(int)
    clin["age"] = pd.to_numeric(clin["age_at_diagnosis"], errors="coerce")
    clin["stage"] = pd.to_numeric(clin["stage"], errors="coerce")

    df = pd.DataFrame({"pid": slides})
    df = df.merge(clin[["pid", "efs_time", "efs_event", "os_time", "os_event", "age", "stage"]], on="pid", how="inner")
    df = df.dropna(subset=["age", "stage"])
    print(f"aligned {len(df)} slides with clinical (age/stage non-missing)")

    # observed S1/S3 (recompute from full-panel matrix for consistency)
    S1 = load_genes(f"{SIG}/S1_progenitor.tsv")
    S3 = load_genes(f"{SIG}/S3_convergent.tsv")
    def score(gs):
        idx = [gene2i[g] for g in gs if g in gene2i]
        if not idx:
            return np.zeros(len(df))
        return mat[[gene2i[s] for s in slides], :][:, idx].mean(axis=1)
    # simpler: mat is slide-major aligned to `slides`
    slide_idx = {s: i for i, s in enumerate(slides)}
    mat_aligned = mat[[slide_idx[p] for p in df["pid"]]]
    def score_aligned(gs):
        idx = [gene2i[g] for g in gs if g in gene2i]
        if not idx:
            return np.zeros(len(df))
        return mat_aligned[:, idx].mean(axis=1)

    obs_s1 = score_aligned(S1)
    obs_s3 = score_aligned(S3)
    for nm, sc in [("S1", obs_s1), ("S3", obs_s3)]:
        df[nm] = (sc - sc.mean()) / sc.std()
    obs_efs_s1 = cox_hr(df, "efs_time", "efs_event", "S1")
    obs_efs_s3 = cox_hr(df, "efs_time", "efs_event", "S3")
    obs_os_s1 = cox_hr(df, "os_time", "os_event", "S1")
    obs_os_s3 = cox_hr(df, "os_time", "os_event", "S3")
    print(f"observed EFS S1 HR={obs_efs_s1[0]:.3f} p={obs_efs_s1[1]:.4g}")
    print(f"observed EFS S3 HR={obs_efs_s3[0]:.3f} p={obs_efs_s3[1]:.4g}")
    print(f"observed OS  S1 HR={obs_os_s1[0]:.3f} p={obs_os_s1[1]:.4g}")
    print(f"observed OS  S3 HR={obs_os_s3[0]:.3f} p={obs_os_s3[1]:.4g}")

    rng = np.random.RandomState(SEED)
    universe = list(range(len(genes)))
    n1 = len(S1); n3 = len(S3)
    null = {"efs_S1": [], "efs_S3": [], "os_S1": [], "os_S3": []}
    for k in range(NPERM):
        pick1 = rng.choice(universe, size=n1, replace=False)
        pick3 = rng.choice(universe, size=n3, replace=False)
        sc1 = mat_aligned[:, pick1].mean(axis=1)
        sc3 = mat_aligned[:, pick3].mean(axis=1)
        df["_r1"] = (sc1 - sc1.mean()) / sc1.std()
        df["_r3"] = (sc3 - sc3.mean()) / sc3.std()
        r = cox_hr(df, "efs_time", "efs_event", "_r1")
        null["efs_S1"].append(r[0] if r else np.nan)
        r = cox_hr(df, "efs_time", "efs_event", "_r3")
        null["efs_S3"].append(r[0] if r else np.nan)
        r = cox_hr(df, "os_time", "os_event", "_r1")
        null["os_S1"].append(r[0] if r else np.nan)
        r = cox_hr(df, "os_time", "os_event", "_r3")
        null["os_S3"].append(r[0] if r else np.nan)
        if (k + 1) % 250 == 0:
            print(f"perm {k+1}/{NPERM} done", flush=True)

    out_rows = []
    for key, obs in [("efs_S1", obs_efs_s1), ("efs_S3", obs_efs_s3),
                     ("os_S1", obs_os_s1), ("os_S3", obs_os_s3)]:
        arr = np.array([x for x in null[key] if not np.isnan(x)])
        # two-sided: how extreme is observed HR relative to null (log scale)
        obs_hr = obs[0]
        # proportion of null with HR <= observed (for HR<1 direction) and >= observed
        p_lo = np.mean(arr <= obs_hr)
        p_hi = np.mean(arr >= obs_hr)
        p_two = 2 * min(p_lo, p_hi)
        out_rows.append({
            "endpoint_signature": key, "obs_HR": round(obs_hr, 3),
            "obs_p": round(obs[1], 4),
            "null_HR_mean": round(float(arr.mean()), 3),
            "null_HR_2.5": round(float(np.percentile(arr, 2.5)), 3),
            "null_HR_97.5": round(float(np.percentile(arr, 97.5)), 3),
            "perm_p_two_sided": round(float(p_two), 4),
        })
        print(f"{key}: obs HR={obs_hr:.3f} | null HR mean={arr.mean():.3f} "
              f"[{np.percentile(arr,2.5):.3f},{np.percentile(arr,97.5):.3f}] | perm p(2-sided)={p_two:.4f}")

    pd.DataFrame(out_rows).to_csv(f"{OUT}/D3_permutation.csv", index=False)
    np.savez(f"{OUT}/D3_permutation_null.npz", **{k: np.array(v) for k, v in null.items()})
    print("saved D3_permutation.csv and D3_permutation_null.npz")


if __name__ == "__main__":
    main()
