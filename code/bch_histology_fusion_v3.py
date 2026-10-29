"""v3 BCH histology x PAX-FOXO1 cross-tab (deduplicated cohort, n=218).

Builds the histology-subtype x fusion-status cross-tab from the source BCH
clinical spreadsheet, applying the same deduplication as dedupe_bch_v3.py
(patient_id = "编号", keep first). Outputs a reviewer-facing JSON and a
supplementary markdown table.
"""

import os, json
import pandas as pd

CHN_XLSX = r"F:\Pathology\Pathology_patch_Normalize_quality\RMS_BCH_Pathology_updated.xlsx"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "outputs", "data_snapshot", "data", "results")
TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "outputs", "data_snapshot", "supplementary", "tables")


def main():
    chn = pd.read_excel(CHN_XLSX, sheet_name="sheet1")
    pid_col = [c for c in chn.columns if "编号" in str(c)][0]
    hist_col = [c for c in chn.columns if "病理类型" in str(c) and "组织" not in str(c)][0]
    fus_col = [c for c in chn.columns if "融合" in str(c)][0]
    chn["pid_str"] = chn[pid_col].astype(int).astype(str).str.zfill(7)
    chn = chn[chn[hist_col].notna() & (chn[hist_col] != "未确")].copy()
    n_raw = len(chn)
    chn = chn.drop_duplicates(subset="pid_str", keep="first")
    chn["hist"] = chn[hist_col].astype(float).astype(int).map({1: "ERMS", 2: "ARMS", 3: "Spindle"})
    chn["fusion"] = chn[fus_col].map({1.0: "Positive", 2.0: "Negative", 3.0: "Not Tested"})

    ctab = chn.groupby(["hist", "fusion"]).size().unstack(fill_value=0)
    order = ["ERMS", "ARMS", "Spindle"]
    ctab = ctab.reindex(order).fillna(0).astype(int)
    for col in ["Positive", "Negative"]:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[["Positive", "Negative"]]
    ctab["Total"] = ctab.sum(axis=1)
    ctab["Fusion-positive (%)"] = (100 * ctab["Positive"] / ctab["Total"]).round(1)

    data = {
        "n_bch_unique": int(len(chn)),
        "n_raw_records": int(n_raw),
        "dedup_note": "218 unique patients after removing one duplicated record (patient 1002311).",
        "histology": {
            h: {"positive": int(ctab.loc[h, "Positive"]),
                "negative": int(ctab.loc[h, "Negative"]),
                "total": int(ctab.loc[h, "Total"]),
                "fusion_positive_pct": float(ctab.loc[h, "Fusion-positive (%)"])}
            for h in order
        },
        "totals": {
            "positive": int(ctab["Positive"].sum()),
            "negative": int(ctab["Negative"].sum()),
            "total": int(ctab["Total"].sum()),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "bch_histology_fusion_v3.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    lines = [
        "# Table S15. BCH histology subtype by PAX-FOXO1 fusion status (n = 218)",
        "",
        "PAX-FOXO1 fusion was confined to the alveolar (ARMS) subtype: 48 of 63 ARMS cases "
        "(76.2%) were fusion-positive, whereas no ERMS (n = 145) or spindle/sclerosing (n = 10) "
        "case was fusion-positive.",
        "",
        "| Histology | Fusion-positive | Fusion-negative | Total | Fusion-positive (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in order:
        lines.append(f"| {h} | {ctab.loc[h,'Positive']} | {ctab.loc[h,'Negative']} | "
                     f"{ctab.loc[h,'Total']} | {ctab.loc[h,'Fusion-positive (%)']} |")
    lines.append(f"| **Total** | {ctab['Positive'].sum()} | {ctab['Negative'].sum()} | "
                 f"{ctab['Total'].sum()} | {round(100*ctab['Positive'].sum()/ctab['Total'].sum(),1)} |")
    lines += [
        "",
        "Abbreviations: ARMS, alveolar rhabdomyosarcoma; BCH, Beijing Children's Hospital; "
        "ERMS, embryonal rhabdomyosarcoma; PAX-FOXO1, PAX3/PAX7-FOXO1 fusion status.",
    ]
    with open(os.path.join(TABLE_DIR, "TableS15_bch_histology_fusion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ctab.to_csv(os.path.join(TABLE_DIR, "TableS15_bch_histology_fusion.csv"), encoding="utf-8-sig")

    print("=== BCH histology x fusion cross-tab (n=%d) ===" % data["n_bch_unique"])
    print(ctab.to_string())
    print("Wrote JSON + TableS15 to submission package.")


if __name__ == "__main__":
    main()
