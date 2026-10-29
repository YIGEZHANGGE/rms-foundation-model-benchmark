# -*- coding: utf-8 -*-
"""v4 survival: adjudicated EFS labels (efs_event_s2), nested scaler/PCA inside folds."""
import os, sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from survival_v3 import (extract_oof_features, run_cox_cv, OUT_DIR, FEAT_DIR, SEED)

COG_TABLE = r"E:/rms_persister/data/cog_clinical/COG_RMS_patient_table.csv"
COHORT = os.path.join(OUT_DIR, "cohort_and_folds.json")

with open(COHORT, encoding="utf-8") as f:
    cohort = json.load(f)
folds, surv, canonical = cohort["folds"], cohort["survival"], cohort["canonical_patient_ids"]

tab = pd.read_csv(COG_TABLE, low_memory=False)
tab = tab.drop_duplicates("patient_id").set_index("patient_id")

rows = []
for p in canonical:
    if p not in tab.index:
        continue
    r = tab.loc[p]
    status = r.get("efs_event_s2")
    days = r.get("efs_days_s3")
    if pd.isna(status) or pd.isna(days):
        continue                      # 状态未裁定 -> 不入组（旧版把它当删失）
    s = surv.get(p, {})
    rows.append({
        "pid": p,
        "efs_time": float(days) / 365.25,
        "efs_event": int(status != "No event"),
        "age": s.get("age"), "stage": s.get("stage"),
        "risk_high": 1 if s.get("risk_group") == "High" else 0,
    })
df = pd.DataFrame(rows)
print("EFS cohort: n=%d, events=%d" % (len(df), int(df.efs_event.sum())), flush=True)
print("status breakdown:", tab.loc[[r['pid'] for r in rows], 'efs_event_s2'].value_counts().to_dict(), flush=True)

feat_files = {os.path.basename(f).replace(".h5", ""): f for f in glob.glob(os.path.join(FEAT_DIR, "*.h5"))}
oof = extract_oof_features(folds, feat_files)
df = df[df.pid.isin(oof)].copy()
print("with OOF features: n=%d, events=%d" % (len(df), int(df.efs_event.sum())), flush=True)
feat_lookup = {p: oof[p] for p in df.pid}

sub = df[df.efs_time > 0].copy()
clin = ["age", "stage", "risk_high"]
subc = sub.dropna(subset=clin).copy()

out = {
    "n_efs": int(len(sub)), "events_efs": int(sub.efs_event.sum()),
    "n_efs_clinical": int(len(subc)),
    "M1_morphology": run_cox_cv(sub, feat_lookup, folds, "efs_time", "efs_event", use_pca=True),
    "M2_clinical": run_cox_cv(subc, feat_lookup, folds, "efs_time", "efs_event", use_pca=False, clin_cols=clin),
    "M3_combined": run_cox_cv(subc, feat_lookup, folds, "efs_time", "efs_event", use_pca=True, clin_cols=clin),
}
with open(os.path.join(OUT_DIR, "survival_results_v4_efs.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
for m in ("M1_morphology", "M2_clinical", "M3_combined"):
    r = out[m]
    print(m, "C=%.3f [%.3f, %.3f] n=%d events=%d" % (r["cv_cindex"], r["cindex_ci"][0], r["cindex_ci"][1], r["n"], r["events"]) if r else "None")
print("=== SURVIVAL V4 DONE ===")
