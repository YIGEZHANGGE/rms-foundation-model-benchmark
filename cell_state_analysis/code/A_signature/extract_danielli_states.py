#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-state marker gene lists from Danielli Supplementary Data 2."""
import os
import csv
import openpyxl

XLSX = r"E:\rms_persister\data\public_sc\danielli_supp\Supplementary_Data_2.xlsx"
OUT = r"E:\rms_persister\results\signatures"
os.makedirs(OUT, exist_ok=True)

wb = openpyxl.load_workbook(XLSX, read_only=True)
ws = wb["RMS atlas - marker genes"]
rows = list(ws.iter_rows(values_only=True))

# locate header row (contains 'Gene', 'Annotation', ...)
header = None
for r in rows[:5]:
    if r and any(c == "Gene" for c in r):
        header = r
        break
cols = {h: i for i, h in enumerate(header) if h}
gene_i = cols["Gene"]
ann_i = cols["Annotation"]
fc_i = cols["Average log2 fold change"]
padj_i = cols["Adjusted P value"]

states = {}
for r in rows[rows.index(header) + 1:]:
    if not r or not r[gene_i]:
        continue
    ann = r[ann_i]
    gene = str(r[gene_i]).strip()
    fc = r[fc_i] if r[fc_i] is not None else 0.0
    padj = r[padj_i] if r[padj_i] is not None else 1.0
    if ann and gene:
        states.setdefault(ann, []).append((gene, fc, padj))

for ann, genes in sorted(states.items()):
    sig = [(g, fc, p) for g, fc, p in genes if p < 0.05]
    sig.sort(key=lambda x: -x[1])
    safe = ann.replace(" ", "_").replace("/", "_").replace("-", "_")
    fp = os.path.join(OUT, f"Danielli_{safe}.tsv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene", "avg_log2FC", "padj"])
        for g, fc, p in sig:
            w.writerow([g, round(fc, 4), f"{p:.3g}"])
    print(f"{ann}: {len(sig)} genes (padj<0.05)")

wb.close()
print("DONE ->", OUT)
