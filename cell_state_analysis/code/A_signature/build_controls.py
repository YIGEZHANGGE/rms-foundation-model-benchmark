#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build S4 (IEG/stress negative control) and S5 (1000 random gene sets)."""
import os
import csv
import random

SIG = r"E:\rms_persister\results\signatures"
os.makedirs(SIG, exist_ok=True)
random.seed(42)

# ---- S4: IEG / dissociation-stress signature (negative control) ----
# Curated from van den Brink dissociation signature + IEG / heat-shock /
# interferon / hypoxia gene families. This is the "S4" control for dissociation
# artefacts (FOS/JUN/HSP/ISG etc.).
IEG = ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "EGR4",
       "NR4A1", "NR4A2", "NR4A3", "ATF3", "ATF4", "DUSP1", "DUSP2", "DUSP4",
       "DUSP5", "DUSP6", "IER2", "IER3", "IER5", "KLF2", "KLF4", "KLF6",
       "KLF10", "ZFP36", "BTG1", "BTG2", "SOCS3", "ARC", "BDNF", "GADD45A",
       "GADD45B", "NFKBIA", "NFKBIZ", "TNFAIP3", "PPP1R15A", "DDIT3", "DDIT4"]
HEAT = ["HSPA1A", "HSPA1B", "HSPA5", "HSPA6", "HSPA8", "HSP90AA1", "HSP90AB1",
        "HSP90B1", "HSPB1", "HSPD1", "DNAJA1", "DNAJB1", "DNAJB4", "DNAJB9",
        "HSPH1", "HSF1", "BAG3", "SERPINH1", "CRYAB", "HSPA4"]
IFN = ["ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI27", "IFI44", "IFI44L",
       "OAS1", "OAS2", "OASL", "MX1", "MX2", "IRF7", "IRF9", "STAT1", "STAT2",
       "RSAD2", "BST2", "IFIH1", "DDX58", "DDX60", "ISG20", "HERC5", "USP18"]
HYPOXIA = ["VEGFA", "SLC2A1", "SLC2A3", "CA9", "LDHA", "NDRG1", "BNIP3",
           "PDK1", "HK1", "HK2", "PGK1", "ENO1", "ALDOA", "TPI1", "GAPDH",
           "ANGPTL4", "ADM", "EGLN3"]

S4 = sorted(set(IEG + HEAT + IFN + HYPOXIA))
with open(os.path.join(SIG, "S4_ieg_stress.tsv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["gene"])
    for g in S4:
        w.writerow([g])
print(f"S4_ieg_stress.tsv: {len(S4)} genes")

# ---- S5: 1000 random gene sets (size-matched to S1, n=297) ----
# Gene universe = union of all genes in Danielli marker tables (approx. RMS-expressed).
import openpyxl
universe = set()
for xlsx, sheet in [(r"E:\rms_persister\data\public_sc\danielli_supp\Supplementary_Data_2.xlsx", "RMS atlas - marker genes")]:
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb[sheet]
    for r in ws.iter_rows(values_only=True):
        if r and r[0] and isinstance(r[0], str) and not r[0].startswith("Gene"):
            universe.add(r[0].strip())
    wb.close()
universe = sorted(u for u in universe if u)
n_genes = 297  # match S1 size
with open(os.path.join(SIG, "S5_random_sets.tsv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["set_id", "genes"])
    for i in range(1000):
        w.writerow([f"rand_{i:04d}", ",".join(random.sample(universe, n_genes))])
print(f"S5_random_sets.tsv: 1000 sets x {n_genes} genes (universe={len(universe)})")
