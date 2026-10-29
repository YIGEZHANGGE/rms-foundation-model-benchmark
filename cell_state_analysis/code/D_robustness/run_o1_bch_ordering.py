#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O1 — BCH (external cohort) S1 ordering external validation.

Trains the encoder -> DeepSpot-M S1 ridge bridge on COG, applies it to BCH
encoder features (mean-pooled), and tests whether the reference
Spindle > ERMS > ARMS ordering is reproduced in the independent BCH cohort.

This is a cross-center transfer of the readout, NOT a direct DeepSpot-M run on
BCH (that is a follow-up once the GPU is free).
"""
import os, glob, json
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEAT = r"F:\Pathology\features"
PRED = r"E:\rms_persister\results\cog_pred\cog_slide_scores.csv"
HIST_COG = r"E:\rms_persister\results\cog_slide_scores_with_histology.csv"
HIST_BCH = r"E:\rms_persister\data\bch_histology.csv"
OUT = r"E:\rms_persister\results\D_robustness"
ENC = ["conch15", "gigapath", "hoptimus0", "phikon", "virchow2"]  # uni2 has no BCH features


def slide_id_from_file(fname, enc):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return stem.replace(f"_{enc}", "")


def load_feats(enc, cohort):
    d = os.path.join(FEAT, f"features_{cohort}_{enc}")
    files = glob.glob(os.path.join(d, "*.h5"))
    X, ids = [], []
    for f in sorted(files):
        sid = slide_id_from_file(f, enc)
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


def main():
    import sys
    pred = pd.read_csv(PRED).rename(columns={"slide": "pid"})
    print("loaded pred", flush=True)
    pred["pid"] = pred["pid"].astype(str)
    hist_cog = pd.read_csv(HIST_COG).rename(columns={"pid": "pid"})
    print("loaded hist_cog", flush=True)
    hist_cog["pid"] = hist_cog["pid"].astype(str)
    hist_map_cog = dict(zip(hist_cog["pid"], hist_cog["histology"]))

    hist_bch = pd.read_csv(HIST_BCH)
    hist_bch["slide"] = hist_bch["slide"].astype(str)
    hist_map_bch = dict(zip(hist_bch["slide"], hist_bch["histology"]))

    ref = pred[pred["pid"].isin(hist_map_cog)]
    y1 = ref["S1"].values.astype(float)

    rows = []
    for enc in ENC:
        print(f"loading cog {enc}...", flush=True)
        Xc, ids_c = load_feats(enc, "cog")
        print(f"  cog {enc}: {Xc.shape}", flush=True)
        id2i_c = {s: i for i, s in enumerate(ids_c)}
        common_c = [p for p in ref["pid"] if p in id2i_c]
        idx_c = [id2i_c[p] for p in common_c]
        Xc_sub = Xc[idx_c]
        yc = ref[ref["pid"].isin(common_c)].set_index("pid").loc[common_c]["S1"].values.astype(float)

        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        mdl.fit(Xc_sub, yc)
        print(f"  fitted ridge {enc}", flush=True)

        print(f"loading chn {enc}...", flush=True)
        Xb, ids_b = load_feats(enc, "chn")
        print(f"  chn {enc}: {Xb.shape}", flush=True)
        yhat_b = mdl.predict(Xb)
        bdf = pd.DataFrame({"slide": ids_b, "S1_bridge": yhat_b})
        bdf["hist"] = bdf["slide"].map(hist_map_bch)
        bdf = bdf.dropna(subset=["hist"])
        order = bdf.groupby("hist")["S1_bridge"].mean()
        present = [h for h in ["Spindle", "ERMS", "ARMS"] if h in order.index]
        vals = [order[h] for h in present]
        mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)) if len(vals) == 3 else False
        rows.append({
            "encoder": enc, "n_bch": len(bdf),
            "Spindle_mean": round(float(order.get("Spindle", np.nan)), 3),
            "ERMS_mean": round(float(order.get("ERMS", np.nan)), 3),
            "ARMS_mean": round(float(order.get("ARMS", np.nan)), 3),
            "Spindle>ERMS>ARMS": mono,
        })
        print(f"{enc}: BCH n={len(bdf)} "
              f"Spindle={order.get('Spindle', np.nan):.3f} ERMS={order.get('ERMS', np.nan):.3f} "
              f"ARMS={order.get('ARMS', np.nan):.3f} mono={mono}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "O1_bch_ordering.csv"), index=False)
    print("\n=== O1 BCH ordering (bridge transfer) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
