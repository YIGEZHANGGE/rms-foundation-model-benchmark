#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D4 — encoder heterogeneity of the H&E -> cell-state readout.

For each benchmark encoder with COG tile features, we:
  1. mean-pool tile features to slide level;
  2. ridge-regress encoder features -> DeepSpot-M S1/S3 (10-fold CV R^2),
     i.e. "consistency with the reference readout" (NOT independent validation:
     the anchor is DeepSpot-M itself);
  3. external-validity check: does the bridged S1 reproduce the reference
     Spindle > ERMS > ARMS ordering;
  4. run the bridged score through the same Cox model (EFS and OS).
"""
import os, glob, json
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.duration.hazard_regression import PHReg

FEAT = r"F:\Pathology\features"
PRED = r"E:\rms_persister\results\cog_pred\cog_slide_scores.csv"
HIST = r"E:\rms_persister\results\cog_slide_scores_with_histology.csv"
CLIN = r"E:\rms_persister\data\cog_clinical\COG_RMS_patient_table.csv"
COHORT = r"C:\Users\onekey\Documents\Codex\2026-08-02\read-project-context-md-in-this\outputs\data_snapshot\data\cohort_and_folds_public.json"
OUT = r"E:\rms_persister\results\D_robustness"
os.makedirs(OUT, exist_ok=True)

ENC = ["conch15", "gigapath", "hoptimus0", "phikon", "uni2", "virchow2"]


def load_slide_feats(enc):
    d = os.path.join(FEAT, f"features_cog_{enc}")
    files = glob.glob(os.path.join(d, "*.h5"))
    X, ids = [], []
    for f in sorted(files):
        sid = os.path.splitext(os.path.basename(f))[0]
        with h5py.File(f, "r") as h:
            key = [k for k in h.keys() if isinstance(h[k], h5py.Dataset)]
            if not key:
                continue
            feats = np.asarray(h[key[0]], dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] == 0:
            continue
        X.append(feats.mean(axis=0))
        ids.append(sid)
    return np.vstack(X).astype(np.float32), ids


def cox_hrp(d, tcol, ecol, scol):
    dd = d[[tcol, ecol, scol, "age", "stage"]].dropna()
    dd = dd[dd[tcol] > 0]
    if dd[ecol].nunique() < 2 or len(dd) < 30:
        return None
    m = PHReg(endog=dd[tcol], exog=dd[[scol, "age", "stage"]], status=dd[ecol], ties="efron")
    r = m.fit()
    return {"n": len(dd), "events": int(dd[ecol].sum()),
            "HR": float(np.exp(float(np.asarray(r.params)[0]))),
            "p": float(np.asarray(r.pvalues)[0])}


def main():
    pred = pd.read_csv(PRED).rename(columns={"slide": "pid"})
    pred["pid"] = pred["pid"].astype(str)
    hist = pd.read_csv(HIST).rename(columns={"pid": "pid"})
    hist["pid"] = hist["pid"].astype(str)
    hist_map = dict(zip(hist["pid"], hist["histology"]))

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

    ref = pred.merge(clin[["pid", "efs_time", "efs_event", "os_time", "os_event", "age", "stage"]], on="pid", how="inner")
    ref["hist"] = ref["pid"].map(hist_map)
    ref = ref.dropna(subset=["hist"])
    print(f"reference slides (in-cohort, with hist): {len(ref)}")

    rows = []
    for enc in ENC:
        X, ids = load_slide_feats(enc)
        id2i = {s: i for i, s in enumerate(ids)}
        common = [p for p in ref["pid"] if p in id2i]
        if len(common) < 50:
            print(f"{enc}: only {len(common)} common slides, skip")
            continue
        idx = [id2i[p] for p in common]
        Xc = X[idx]
        sub = ref[ref["pid"].isin(common)].set_index("pid").loc[common].reset_index()
        y1 = sub["S1"].values.astype(float)
        y3 = sub["S3"].values.astype(float)

        # 10-fold CV bridge R^2
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        def bridge_r2(y):
            mdl = make_pipeline(StandardScaler(),
                                RidgeCV(alphas=np.logspace(-2, 4, 25)))
            yhat = cross_val_predict(mdl, Xc, y, cv=kf)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1 - ss_res / ss_tot, yhat
        r2_1, yhat1 = bridge_r2(y1)
        r2_3, yhat3 = bridge_r2(y3)

        # external validity: Spindle > ERMS > ARMS for bridged S1
        sub = sub.copy()
        sub["yhat1"] = yhat1
        order = sub.groupby("hist")["yhat1"].mean()
        full = ["Spindle", "ERMS", "ARMS"]
        present = [h for h in full if h in order.index]
        vals = [order[h] for h in present]
        mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

        # Cox on bridged scores (z-scored within cohort)
        for sname, yh in [("S1_bridge", yhat1), ("S3_bridge", yhat3)]:
            sub[sname] = (yh - yh.mean()) / yh.std()
        efs1 = cox_hrp(sub, "efs_time", "efs_event", "S1_bridge")
        efs3 = cox_hrp(sub, "efs_time", "efs_event", "S3_bridge")
        os1 = cox_hrp(sub, "os_time", "os_event", "S1_bridge")
        os3 = cox_hrp(sub, "os_time", "os_event", "S3_bridge")

        dim = Xc.shape[1]
        rows.append({
            "encoder": enc, "dim": dim, "n_slides": len(common),
            "bridge_R2_S1": round(r2_1, 3), "bridge_R2_S3": round(r2_3, 3),
            "Spindle>ERMS>ARMS": mono,
            "order": " > ".join(present),
            "S1_means": {h: round(float(order[h]), 3) for h in present},
            "EFS_S1_HR": None if efs1 is None else round(efs1["HR"], 3),
            "EFS_S1_p": None if efs1 is None else round(efs1["p"], 4),
            "EFS_S3_HR": None if efs3 is None else round(efs3["HR"], 3),
            "EFS_S3_p": None if efs3 is None else round(efs3["p"], 4),
            "OS_S1_HR": None if os1 is None else round(os1["HR"], 3),
            "OS_S1_p": None if os1 is None else round(os1["p"], 4),
            "OS_S3_HR": None if os3 is None else round(os3["HR"], 3),
            "OS_S3_p": None if os3 is None else round(os3["p"], 4),
        })
        print(f"{enc}: dim={dim} n={len(common)} "
              f"R2_S1={r2_1:.3f} R2_S3={r2_3:.3f} mono={mono} ({order[present[0]]:.3f}>{order[present[1]]:.3f}>{order[present[2]]:.3f}) "
              f"EFS_S1_HR={None if efs1 is None else efs1['HR']:.3f} "
              f"OS_S1_HR={None if os1 is None else os1['HR']:.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT, "D4_encoder_heterogeneity.csv"), index=False)
    # print full table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print("\n=== D4 summary table ===")
    print(out_df[["encoder", "dim", "n_slides", "bridge_R2_S1", "bridge_R2_S3",
                  "Spindle>ERMS>ARMS", "EFS_S1_HR", "EFS_S1_p",
                  "OS_S1_HR", "OS_S1_p", "OS_S3_HR", "OS_S3_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
