#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I2 — attention concentration x cell-state scores (BCH, zero new data).

Does a higher progenitor (S1) or convergent (S3) tumor rely on diffuse or
concentrated diagnostic attention? Correlates slide-level attention entropy /
top-10 mass (CONCH, BCH) with DeepSpot-M S1/S3.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

ATTN = r"C:\Users\onekey\Documents\Codex\2026-08-02\read-project-context-md-in-this\outputs\data_snapshot\data\results\attention_summary_v3.json"
BCH_SCORE = r"E:\rms_persister\results\D_robustness\bch_slide_scores.csv"
OUT = r"E:\rms_persister\results\D_robustness"


def main():
    attn = json.load(open(ATTN, encoding="utf-8"))["per_patient"]
    adf = pd.DataFrame(attn)
    adf["patient_id"] = adf["patient_id"].astype(str)
    b = pd.read_csv(BCH_SCORE)
    b["slide"] = b["slide"].astype(str)
    m = adf.merge(b, left_on="patient_id", right_on="slide", how="inner")
    print(f"merged {len(m)} BCH patients (attention x cell-state)")

    rows = []
    for score in ["S1", "S3", "S4"]:
        for attn_metric in ["entropy", "top10_attn"]:
            r, p = stats.spearmanr(m[score], m[attn_metric])
            rp, pp = stats.pearsonr(m[score], m[attn_metric])
            rows.append({
                "score": score, "attn_metric": attn_metric,
                "spearman_r": round(float(r), 3), "spearman_p": round(float(p), 4),
                "pearson_r": round(float(rp), 3), "pearson_p": round(float(pp), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/I2_attention_cellstate.csv", index=False)
    print(out.to_string(index=False))

    # by histology attention
    print("\nattention by histology:")
    print(m.groupby("histology")[["entropy", "top10_attn"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
