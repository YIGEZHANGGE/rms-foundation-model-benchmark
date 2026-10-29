#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1-T3 fusion-axis tests.

T1: COG S1, ARMS vs {ERMS+Spindle} (histology proxy for fusion).
T2: BCH S1, fusion+ (48) vs fusion- (170) (true fusion labels).
T3: BCH S1, ARMS vs {ERMS+Spindle} (histology, comparable to T1).

Each: Mann-Whitney U + 10,000 bootstrap 95% CI on mean difference (seed 42).
"""
import numpy as np
import pandas as pd
from scipy import stats

COG_SCORE = r"E:\rms_persister\results\cog_pred\cog_slide_scores.csv"
COG_HIST = r"E:\rms_persister\results\cog_slide_scores_with_histology.csv"
BCH_SCORE = r"E:\rms_persister\results\D_robustness\bch_slide_scores.csv"
BCH_FUS = r"E:\rms_persister\data\bch_histology_fusion.csv"
BCH_HIST = r"E:\rms_persister\data\bch_histology.csv"
OUT = r"E:\rms_persister\results\D_robustness"
NBOOT = 10000
SEED = 42


def run_test(a, b, label_a, label_b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff = a.mean() - b.mean()
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(NBOOT):
        sa = rng.choice(a, len(a), replace=True)
        sb = rng.choice(b, len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pooled_sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    d = diff / pooled_sd
    return {
        "a": label_a, "n_a": len(a), "mean_a": round(float(a.mean()), 3),
        "b": label_b, "n_b": len(b), "mean_b": round(float(b.mean()), 3),
        "mean_diff": round(float(diff), 3),
        "bootstrap_95CI": f"[{lo:.3f}, {hi:.3f}]",
        "MW_p": round(float(p), 4),
        "cohen_d": round(float(d), 3),
    }


def main():
    cog = pd.read_csv(COG_SCORE).rename(columns={"slide": "pid"})
    cog_hist = pd.read_csv(COG_HIST).rename(columns={"pid": "pid"})
    cog = cog.merge(cog_hist[["pid", "histology"]], on="pid", how="inner")

    bch = pd.read_csv(BCH_SCORE)
    bch["slide"] = bch["slide"].astype(str)
    fus = pd.read_csv(BCH_FUS); fus["pid_str"] = fus["pid_str"].astype(str)
    hist = pd.read_csv(BCH_HIST); hist["slide"] = hist["slide"].astype(str)
    bch = bch.merge(fus[["pid_str", "hist", "fusion"]], left_on="slide", right_on="pid_str", how="inner")
    bch = bch.merge(hist[["slide", "histology"]], on="slide", how="left")

    # T1: COG ARMS vs {ERMS+Spindle}
    arms_c = cog[cog["histology"] == "ARMS"]["S1"]
    rest_c = cog[cog["histology"].isin(["ERMS", "Spindle"])]["S1"]
    t1 = run_test(arms_c, rest_c, "ARMS", "ERMS+Spindle")
    t1["test"] = "T1 (COG histology)"

    # T2: BCH fusion+ vs fusion-
    fp = bch[bch["fusion"] == "Positive"]["S1"]
    fn = bch[bch["fusion"] == "Negative"]["S1"]
    t2 = run_test(fp, fn, "fusion+", "fusion-")
    t2["test"] = "T2 (BCH fusion)"

    # T3: BCH ARMS vs {ERMS+Spindle}
    arms_b = bch[bch["histology"] == "ARMS"]["S1"]
    rest_b = bch[bch["histology"].isin(["ERMS", "Spindle"])]["S1"]
    t3 = run_test(arms_b, rest_b, "ARMS", "ERMS+Spindle")
    t3["test"] = "T3 (BCH histology)"

    out = pd.DataFrame([t1, t2, t3])
    out.to_csv(f"{OUT}/T1T3_fusion_axis.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
