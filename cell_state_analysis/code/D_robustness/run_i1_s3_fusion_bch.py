#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I1 — S3 (convergent state) x PAX-FOXO1 fusion association in BCH.

Trains the encoder -> DeepSpot-M S3 bridge on COG, applies to BCH, and tests
whether the bridged S3 differs between fusion-positive ARMS (n=48) and
fusion-negative ARMS (n=15), cross-validating Whitfield's claim that the
convergent state is independent of fusion status.

Bridge caveat: cross-center transfer, domain shift applies.
"""
import os, glob
import numpy as np
import pandas as pd
import h5py
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEAT = r"F:\Pathology\features"
PRED = r"E:\rms_persister\results\cog_pred\cog_slide_scores.csv"
HIST_COG = r"E:\rms_persister\results\cog_slide_scores_with_histology.csv"
BCH_FUS = r"E:\rms_persister\data\bch_histology_fusion.csv"
OUT = r"E:\rms_persister\results\D_robustness"
ENC = ["conch15", "gigapath", "hoptimus0", "phikon", "virchow2"]


def slide_id_from_file(fname, enc):
    return os.path.splitext(os.path.basename(fname))[0].replace(f"_{enc}", "")


def load_feats(enc, cohort):
    d = os.path.join(FEAT, f"features_{cohort}_{enc}")
    X, ids = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.h5"))):
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
    pred = pd.read_csv(PRED).rename(columns={"slide": "pid"})
    pred["pid"] = pred["pid"].astype(str)
    hist_cog = pd.read_csv(HIST_COG).rename(columns={"pid": "pid"})
    hist_cog["pid"] = hist_cog["pid"].astype(str)
    hist_map_cog = dict(zip(hist_cog["pid"], hist_cog["histology"]))

    bch = pd.read_csv(BCH_FUS)
    bch["pid_str"] = bch["pid_str"].astype(str)
    arms = bch[bch["hist"] == "ARMS"].copy()
    print(f"BCH ARMS: {len(arms)} (fusion+ {sum(arms['fusion']=='Positive')}, fusion- {sum(arms['fusion']=='Negative')})", flush=True)
    fus_map = dict(zip(arms["pid_str"], arms["fusion"]))

    ref = pred[pred["pid"].isin(hist_map_cog)]
    y3 = ref["S3"].values.astype(float)

    rows = []
    for enc in ENC:
        print(f"loading {enc}...", flush=True)
        Xc, ids_c = load_feats(enc, "cog")
        id2i_c = {s: i for i, s in enumerate(ids_c)}
        common_c = [p for p in ref["pid"] if p in id2i_c]
        idx_c = [id2i_c[p] for p in common_c]
        yc = ref[ref["pid"].isin(common_c)].set_index("pid").loc[common_c]["S3"].values.astype(float)
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        mdl.fit(Xc[idx_c], yc)

        Xb, ids_b = load_feats(enc, "chn")
        yhat = mdl.predict(Xb)
        bdf = pd.DataFrame({"slide": ids_b, "S3_bridge": yhat})
        bdf["fusion"] = bdf["slide"].map(fus_map)
        bdf = bdf.dropna(subset=["fusion"])
        pos = bdf[bdf["fusion"] == "Positive"]["S3_bridge"]
        neg = bdf[bdf["fusion"] == "Negative"]["S3_bridge"]
        if len(pos) >= 3 and len(neg) >= 3:
            u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        else:
            u, p = np.nan, np.nan
        rows.append({
            "encoder": enc, "n_pos": len(pos), "n_neg": len(neg),
            "S3_pos_mean": round(float(pos.mean()), 3),
            "S3_neg_mean": round(float(neg.mean()), 3),
            "MW_p": round(float(p), 4),
        })
        print(f"  {enc}: pos({len(pos)})={pos.mean():.3f} vs neg({len(neg)})={neg.mean():.3f} MW p={p:.4g}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "I1_s3_fusion_bch.csv"), index=False)
    print("\n=== I1 S3 x fusion (BCH ARMS) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
