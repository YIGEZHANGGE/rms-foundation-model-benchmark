"""v3 Table 1: frozen n=390 cohort + BCH 219."""

import os, json, re
import numpy as np
import pandas as pd

PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")
COHORT_PATH = os.path.join(OUT_DIR, "cohort_and_folds.json")
COG_CSV = r"C:\Users\onekey\WorkBuddy\2026-05-07-task-1\COG_RMS_patient_table.csv"
CHN_XLSX = r"F:\Pathology\Pathology_patch_Normalize_quality\RMS_BCH_Pathology_updated.xlsx"

with open(COHORT_PATH, "r", encoding="utf-8") as f:
    cohort = json.load(f)
canonical = cohort["canonical_patient_ids"]
surv = cohort["survival"]

cog = pd.read_csv(COG_CSV).set_index("patient_id")
chn = pd.read_excel(CHN_XLSX)


def parse_bch_age(s):
    if pd.isna(s) or not isinstance(s, str):
        return np.nan
    s = s.strip()
    years = months = days = 0
    m = re.search(r"(\d+)\s*岁", s)
    if m:
        years = int(m.group(1))
    m = re.search(r"(\d+)\s*月", s)
    if m:
        months = int(m.group(1))
    m = re.search(r"(\d+)\s*天", s) or re.search(r"(\d+)\s*小时", s)
    if m:
        days = int(m.group(1))
    return years + months / 12 + days / 365.25


chn["age_years"] = chn.iloc[:, 10].apply(parse_bch_age)
hist_col = [c for c in chn.columns if "病理类型" in str(c) and "组织" not in str(c)][0]
irs_col = [c for c in chn.columns if "IRS" in str(c)][0]
fusion_col = [c for c in chn.columns if "融合" in str(c)][0]
spec_col = [c for c in chn.columns if "穿刺" in str(c) or "活检" in str(c)][0]

chn = chn[chn[hist_col].notna() & (chn[hist_col] != "未确")]
chn["hist"] = chn[hist_col].astype(float).astype(int).map({1: "ERMS", 2: "ARMS", 3: "Spindle"})
chn["stage"] = chn[irs_col]
chn["fusion"] = chn[fusion_col].map({1.0: "Positive", 2.0: "Negative", 3.0: "Not Tested"})
chn["specimen"] = chn[spec_col].map({1.0: "Puncture", 2.0: "Biopsy"})
chn["male"] = chn.iloc[:, 8].astype(str).str.strip().isin(["男", "男性", "M", "Male"])


def n_pct(n, total):
    return "—" if total == 0 else f"{n} ({100 * n / total:.1f}%)"


def med_range(s, decimals=1):
    clean = s.dropna()
    if len(clean) == 0:
        return "—"
    return f"{clean.median():.{decimals}f} [{clean.min():.{decimals}f}–{clean.max():.{decimals}f}]"


n_cog = len(canonical)
n_chn = len(chn)

hist_counts = {0: 0, 1: 0, 2: 0}
stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}
stage_missing = 0
for p in canonical:
    hist_counts[cohort["labels"][p]] += 1
    st = surv[p]["stage"]
    if st is None:
        stage_missing += 1
    else:
        stage_counts[int(st)] += 1

rows = []
def add(label, cog_val, chn_val):
    rows.append({"Characteristic": label, f"COG (n={n_cog})": cog_val, f"BCH (n={n_chn})": chn_val})

add("No. of patients", str(n_cog), str(n_chn))
add("Age, years", med_range(cog.loc[canonical, "age_at_diagnosis"]), med_range(chn["age_years"], 2))
add("Gender", "", "")
add("  Male", n_pct(int(cog.loc[canonical, "gender"].eq("Male").sum()), n_cog),
    n_pct(int(chn["male"].sum()), n_chn))
add("  Female", n_pct(int(cog.loc[canonical, "gender"].eq("Female").sum()), n_cog),
    n_pct(int((~chn["male"]).sum()), n_chn))
add("", "", "")
add("Histological Subtype", "", "")
for cls, name in [(0, "ERMS"), (1, "ARMS"), (2, "Spindle")]:
    add(f"  {name}", n_pct(hist_counts[cls], n_cog), n_pct(int((chn["hist"] == name).sum()), n_chn))
add("", "", "")
add("IRS Clinical Stage", "", "")
for s, name in [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]:
    add(f"  Stage {name}", n_pct(stage_counts[s], n_cog - stage_missing),
        n_pct(int((chn["stage"] == float(s)).sum()), int(chn["stage"].notna().sum())))
add("  Not Reported", n_pct(stage_missing, n_cog), n_pct(int(chn["stage"].isna().sum()), n_chn))
add(f"  (Total with staging)", f"n={n_cog - stage_missing}", f"n={int(chn['stage'].notna().sum())}")
add("", "", "")
add("PAX-FOXO1 Fusion", "", "")
add("  Positive", "—", n_pct(int((chn["fusion"] == "Positive").sum()), n_chn))
add("  Negative", "—", n_pct(int((chn["fusion"] == "Negative").sum()), n_chn))
add("", "", "")
add("Specimen Type", "", "")
for sp in ["Puncture", "Biopsy"]:
    add(f"  {sp}", "—", n_pct(int((chn["specimen"] == sp).sum()), n_chn))
add("", "", "")
add("Survival Events", "", "")
add("  EFS events", n_pct(sum(surv[p]["efs_event"] for p in canonical), n_cog), "—")
add("  OS events", n_pct(sum(surv[p]["os_event"] for p in canonical), n_cog), "—")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "table1_v3.csv"), index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

# Adult sensitivity info
ages = cog.loc[canonical, "age_at_diagnosis"].dropna()
print(f"\nCOG age: median={ages.median():.1f}, range={ages.min():.1f}-{ages.max():.1f}")
print(f"Patients >18y: {(ages > 18).sum()}, >21y: {(ages > 21).sum()}, >30y: {(ages > 30).sum()}")
print("=== TABLE1 V3 DONE ===")
