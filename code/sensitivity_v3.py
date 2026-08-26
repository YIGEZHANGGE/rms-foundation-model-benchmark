"""v3 Sensitivity: McNemar+Bonferroni for encoders and adult-exclusion analysis."""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score

PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")
COHORT_PATH = os.path.join(OUT_DIR, "cohort_and_folds.json")
ENCODERS = ["conch", "phikon", "hoptimus0", "gigapath"]

with open(COHORT_PATH, "r", encoding="utf-8") as f:
    cohort = json.load(f)
age = {str(p): cohort["survival"][p]["age"] for p in cohort["canonical_patient_ids"]}


def load_oof(enc):
    d = pd.read_csv(os.path.join(OUT_DIR, f"diagnosis_oof_{enc}.csv"))
    d["age"] = d["patient_id"].astype(str).map(age)
    return d


def two_class(t, p):
    t2 = [0 if x in (0, 2) else 1 for x in t]
    p2 = [0 if x in (0, 2) else 1 for x in p]
    return t2, p2


def metrics(t, p, binary=False):
    if binary:
        t, p = two_class(t, p)
    return {
        "n": int(len(t)),
        "acc": float(accuracy_score(t, p)),
        "f1m": float(f1_score(t, p, average="macro", zero_division=0)),
    }


def mcnemar(t, p1, p2):
    e01 = int(((np.array(p1) != np.array(t)) & (np.array(p2) == np.array(t))).sum())
    e10 = int(((np.array(p1) == np.array(t)) & (np.array(p2) != np.array(t))).sum())
    if e01 + e10 == 0:
        return 1.0
    stat = (abs(e01 - e10) - 1) ** 2 / (e01 + e10)
    return float(1 - chi2.cdf(stat, 1))


def pairwise_mcnemar(ds, binary=False):
    pairs = {}
    ts = {}
    for enc in ENCODERS:
        t = ds[enc]["true"].values
        p = ds[enc]["pred"].values
        if binary:
            t, p = two_class(t, p)
        ts[enc] = (t, p)
    k = len(ENCODERS) * (len(ENCODERS) - 1) // 2
    for i in range(len(ENCODERS)):
        for j in range(i + 1, len(ENCODERS)):
            a, b = ENCODERS[i], ENCODERS[j]
            raw = mcnemar(ts[a][0], ts[a][1], ts[b][1])
            pairs[f"{a}_vs_{b}"] = {
                "raw_p": raw,
                "adjusted_p": min(1.0, raw * k),
                "k": k,
            }
    return pairs


def main():
    ds = {enc: load_oof(enc) for enc in ENCODERS}
    results = {
        "meta": {
            "generated_by": os.path.basename(__file__),
            "encoders": ENCODERS,
            "adult_cutoff": 21,
            "note": "Adult-exclusion sensitivity is evaluation-level: models were trained on the full frozen cohort; predictions are filtered by age. Full retraining without adults remains a possible follow-up.",
        },
        "mcnemar_3class": pairwise_mcnemar(ds, binary=False),
        "mcnemar_2class": pairwise_mcnemar(ds, binary=True),
        "adult_sensitivity": {},
    }

    for enc, d in ds.items():
        all_3 = metrics(d["true"], d["pred"])
        all_2 = metrics(d["true"], d["pred"], binary=True)
        le21 = d[d["age"] <= 21]
        le21_3 = metrics(le21["true"], le21["pred"])
        le21_2 = metrics(le21["true"], le21["pred"], binary=True)
        le18 = d[d["age"] <= 18]
        le18_3 = metrics(le18["true"], le18["pred"])
        le18_2 = metrics(le18["true"], le18["pred"], binary=True)
        gt21 = d[d["age"] > 21]
        gt21_3 = metrics(gt21["true"], gt21["pred"])
        gt21_2 = metrics(gt21["true"], gt21["pred"], binary=True)
        results["adult_sensitivity"][enc] = {
            "all": {"n": all_3["n"], "acc3": all_3["acc"], "acc2": all_2["acc"], "f1m3": all_3["f1m"], "f1m2": all_2["f1m"]},
            "age_le21": {"n": le21_3["n"], "acc3": le21_3["acc"], "acc2": le21_2["acc"], "f1m3": le21_3["f1m"], "f1m2": le21_2["f1m"]},
            "age_le18": {"n": le18_3["n"], "acc3": le18_3["acc"], "acc2": le18_2["acc"], "f1m3": le18_3["f1m"], "f1m2": le18_2["f1m"]},
            "age_gt21": {"n": gt21_3["n"], "acc3": gt21_3["acc"], "acc2": gt21_2["acc"], "f1m3": gt21_3["f1m"], "f1m2": gt21_2["f1m"]},
        }

    with open(os.path.join(OUT_DIR, "diagnosis_stats_v3.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = ["# v3 Sensitivity Analysis", ""]
    lines.append("## McNemar (3-class, 4 encoders, k=6)")
    lines.append("| Pair | raw p | adjusted p |")
    lines.append("|---|---|---|")
    for k, v in results["mcnemar_3class"].items():
        lines.append(f"| {k} | {v['raw_p']:.4f} | {v['adjusted_p']:.4f} |")
    lines.append("")
    lines.append("## McNemar (2-class, k=6)")
    lines.append("| Pair | raw p | adjusted p |")
    lines.append("|---|---|---|")
    for k, v in results["mcnemar_2class"].items():
        lines.append(f"| {k} | {v['raw_p']:.4f} | {v['adjusted_p']:.4f} |")
    lines.append("")
    lines.append("## Adult-exclusion sensitivity (evaluation-level)")
    lines.append("")
    lines.append("| Encoder | Group | n | Acc 3-class | F1m 3-class | Acc 2-class | F1m 2-class |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for enc, r in results["adult_sensitivity"].items():
        for grp in ["all", "age_le21", "age_le18", "age_gt21"]:
            x = r[grp]
            lines.append(
                f"| {enc} | {grp} | {x['n']} | {x['acc3']:.4f} | {x['f1m3']:.4f} | {x['acc2']:.4f} | {x['f1m2']:.4f} |"
            )
    lines.append("")
    lines.append("> Evaluation-level sensitivity: existing OOF predictions filtered by age; models still trained on the full cohort.")
    with open(os.path.join(OUT_DIR, "SENSITIVITY_V3.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("=== SENSITIVITY V3 DONE ===")


if __name__ == "__main__":
    main()
