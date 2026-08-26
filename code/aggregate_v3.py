"""Aggregate all v3 JSONs into SUMMARY_V3.md."""

import os, json, glob
import pandas as pd

PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")

def load(name):
    p = os.path.join(OUT_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

lines = ["# RMS v3 Summary", ""]
lines.append("> **Data note:** CONCH and CONCH 1.5 features are duplicates/almost identical in this dataset; CONCH 1.5 is excluded from the final summary.")
lines.append("")

diag = load("diagnosis_results_v3.json")
if diag:
    lines.append("## Diagnosis (5 encoders, frozen n=390)")
    lines.append("")
    lines.append("| Encoder | Accuracy | F1-macro | ERMS F1 | ARMS F1 | Spindle F1 |")
    lines.append("|---|---|---|---|---|---|")
    for enc, r in diag.items():
        if enc == "conch15":
            continue
        pc = r.get("per_class", {})
        lines.append(
            f"| {enc} | {r['acc_mean']:.4f}±{r['acc_std']:.4f} | {r['f1m_mean']:.4f}±{r['f1m_std']:.4f} "
            f"| {pc.get('ERMS',{}).get('mean',float('nan')):.4f} | {pc.get('ARMS',{}).get('mean',float('nan')):.4f} "
            f"| {pc.get('Spindle',{}).get('mean',float('nan')):.4f} |"
        )
    lines.append("")

abl = load("architecture_ablation_v3.json")
if abl:
    lines.append("## Architecture ablation (3 encoders × 4 custom variants)")
    lines.append("")
    lines.append("| Encoder | Architecture | Accuracy | F1-macro |")
    lines.append("|---|---|---|---|")
    for enc, archs in abl.items():
        if enc == "conch15":
            continue
        for arch, r in archs.items():
            lines.append(f"| {enc} | {arch} | {r['acc_mean']:.4f}±{r['acc_std']:.4f} | {r['f1m_mean']:.4f}±{r['f1m_std']:.4f} |")
    lines.append("")

ext = load("external_validation_v3.json")
if ext:
    lines.append("## External validation (BCH n=219)")
    lines.append("")
    lines.append("| Encoder | Accuracy | 95% CI | F1-macro | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for enc, r in ext.items():
        if enc == "conch15":
            continue
        lines.append(f"| {enc} | {r['acc']:.4f} | {r['acc_ci'][0]:.4f}-{r['acc_ci'][1]:.4f} | {r['f1m']:.4f} | {r['f1m_ci'][0]:.4f}-{r['f1m_ci'][1]:.4f} |")
    lines.append("")

fox = load("foxo1_results_v3.json")
if fox:
    lines.append("## FOXO1 (BCH, exploratory)")
    lines.append("")
    lines.append("| Encoder | AUC | Accuracy | Sensitivity | Specificity |")
    lines.append("|---|---|---|---|---|")
    for enc, r in fox.items():
        if enc == "conch15":
            continue
        lines.append(f"| {enc} | {r['auc']:.4f} | {r['acc']:.4f} | {r['sens']:.4f} | {r['spec']:.4f} |")
    lines.append("")

surv = load("survival_results_v3.json")
if surv:
    lines.append("## Survival (true OOF, CONCH 1.5)")
    lines.append("")
    for outcome in ["efs", "os"]:
        if outcome not in surv:
            continue
        lines.append(f"### {outcome.upper()}")
        lines.append("")
        lines.append("| Model | CV C-index | 95% CI | Events | 1y AUC | 3y AUC | 5y AUC |")
        lines.append("|---|---|---|---|---|---|---|")
        for label, r in surv[outcome].items():
            if not r:
                lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            ta = r.get("time_auc", {})
            ta = {str(k): v for k, v in ta.items()}
            lines.append(
                f"| {label} | {r['cv_cindex']:.4f} | {r['cindex_ci'][0]:.4f}-{r['cindex_ci'][1]:.4f} | {r['events']} "
                f"| {ta.get('1', float('nan')):.3f} | {ta.get('3', float('nan')):.3f} | {ta.get('5', float('nan')):.3f} |"
            )
        lines.append("")

with open(os.path.join(OUT_DIR, "SUMMARY_V3.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("=== AGGREGATE V3 DONE ===")
