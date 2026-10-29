"""Compare ablation run 1 (seed 42) and run 2 (seed 2026) for run-to-run stability.

Outputs a reviewer-facing summary JSON (ablation_run_stability_v3.json) and a
supplementary markdown table (TableS16_ablation_run_stability.md).
"""

import os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(ROOT, "outputs", "data_snapshot", "data", "results")
TAB_DIR = os.path.join(ROOT, "outputs", "data_snapshot", "supplementary", "tables")
RUN1 = os.path.join(RES_DIR, "architecture_ablation_v3.json")
RUN2 = os.path.join(RES_DIR, "architecture_ablation_v3_run2.json")


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    r1 = load(RUN1)
    r2 = load(RUN2)
    models = ["conch", "hoptimus0"]
    archs = ["AttentionMIL", "MultiClusterAttention", "TransformerInstanceAttention", "TransformerPooling"]
    rows = []
    summary = {"models": {}, "meta": {
        "run1_seed": 42, "run2_seed": 2026,
        "note": "Same frozen n=390 cohort, same five stratified folds, same training schedule; "
                "only the training seed differs. Run-to-run fold-level mean differences and the "
                "architectural ranking are reported."}}
    for m in models:
        m_sum = {}
        for a in archs:
            d1 = r1[m][a]; d2 = r2[m][a]
            a1 = np.array([f["acc"] for f in d1["folds"]])
            a2 = np.array([f["acc"] for f in d2["folds"]])
            diff = float(np.mean(a2) - np.mean(a1))
            maxdiff = float(np.max(np.abs(a2 - a1)))
            # paired across folds
            paired_diff = a2 - a1
            m_sum[a] = {"run1_acc_mean": float(d1["acc_mean"]), "run1_acc_std": float(d1["acc_std"]),
                        "run2_acc_mean": float(d2["acc_mean"]), "run2_acc_std": float(d2["acc_std"]),
                        "mean_diff_run2_minus_run1": round(diff, 4),
                        "max_fold_abs_diff": round(maxdiff, 4),
                        "run1_f1m_mean": float(d1["f1m_mean"]), "run2_f1m_mean": float(d2["f1m_mean"])}
            rows.append([m, a, round(float(d1["acc_mean"]), 3), round(float(d1["acc_std"]), 3),
                         round(float(d2["acc_mean"]), 3), round(float(d2["acc_std"]), 3),
                         round(diff, 3), round(maxdiff, 3)])
        # rank by run1 acc and run2 acc
        rank1 = sorted(archs, key=lambda a: r1[m][a]["acc_mean"], reverse=True)
        rank2 = sorted(archs, key=lambda a: r2[m][a]["acc_mean"], reverse=True)
        m_sum["_rank_run1"] = rank1
        m_sum["_rank_run2"] = rank2
        m_sum["_rank_consistent"] = rank1 == rank2
        summary["models"][m] = m_sum

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "ablation_run_stability_v3.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    lines = [
        "# Supplementary table: architecture-ablation run-to-run stability (COG, n = 390)",
        "",
        "The architecture ablation was repeated with a different training seed (seed 42 vs seed 2026) "
        "on the same frozen cohort and the same five stratified folds, with the same training schedule. "
        "Mean fold-level accuracy differences between runs were small (up to 0.033), but the ranking "
        "between AttentionMIL and MultiClusterAttention reversed for H-optimus-0 (run 1 favored "
        "MultiClusterAttention, run 2 favored AttentionMIL), whereas the CONCH ranking was unchanged. "
        "TransformerPooling was at the majority-class baseline in run 1 (0.749) but did not collapse in "
        "run 2 (0.782 CONCH; 0.756 H-optimus-0). The best-pooling choice is therefore seed-sensitive and "
        "model-specific, and the ablation is read as a map of the pooling-architecture search space rather "
        "than a recommendation of a single variant.",
        "",
        "| Model | Variant | Run1 acc | Run1 std | Run2 acc | Run2 std | Mean diff | Max fold diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += ["", "Run 1 and run 2 used the same folds, labels, and training hyperparameters; only the "
              "random seed differed. Std is the standard deviation across the five folds.",
              "", "Abbreviations: COG, Children's Oncology Group."]
    with open(os.path.join(TAB_DIR, "TableS16_ablation_run_stability.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=== Run-to-run stability (run1 seed42 vs run2 seed2026) ===")
    for m in models:
        print(f"\n[{m}]")
        for a in archs:
            s = summary["models"][m][a]
            print(f"  {a}: run1={s['run1_acc_mean']:.3f} run2={s['run2_acc_mean']:.3f} "
                  f"diff={s['mean_diff_run2_minus_run1']:+.3f} maxfold={s['max_fold_abs_diff']:.3f}")
        print(f"  rank run1: {summary['models'][m]['_rank_run1']}")
        print(f"  rank run2: {summary['models'][m]['_rank_run2']}  consistent={summary['models'][m]['_rank_consistent']}")
    print("Wrote ablation_run_stability_v3.json + TableS16")


if __name__ == "__main__":
    main()
