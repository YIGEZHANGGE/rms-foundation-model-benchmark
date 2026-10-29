#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tile-level variance decomposition: within-slide vs between-slide S1/S3 variance."""
import glob
import os
import csv
import random
import numpy as np
import torch
from PIL import Image
from deepspotm import DeepSpotM

WEIGHTS = r"E:\rms_persister\src\C2_zeroshot\DeepSpotM_weights"
SIG = r"E:\rms_persister\results\signatures"
PATCHES = r"F:\Pathology\IDC_RMS_patches"
random.seed(42)


def load_genes(fn):
    g = []
    with open(os.path.join(SIG, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            g.append(r["gene"])
    return g


S1 = load_genes("S1_progenitor.tsv")
S3 = load_genes("S3_convergent.tsv")
ALL = sorted(set(S1 + S3))

model, ip = DeepSpotM.from_pretrained(WEIGHTS, source="scgpt")
model = model.eval()
device = next(model.parameters()).device
mg = set(model.gene_names)
QUERY = [g for g in ALL if g in mg]
S1i = [QUERY.index(g) for g in S1 if g in QUERY]
S3i = [QUERY.index(g) for g in S3 if g in QUERY]

slides = sorted([d for d in os.listdir(PATCHES) if os.path.isdir(os.path.join(PATCHES, d))])
slides = random.sample(slides, 25)

rows = []
for slide in slides:
    files = sorted(glob.glob(os.path.join(PATCHES, slide, "*.jpg")))
    if len(files) > 400:
        files = random.sample(files, 400)
    for i in range(0, len(files), 64):
        batch = files[i:i + 64]
        tiles = []
        for f in batch:
            im = Image.open(f).convert("RGB")
            w, h = im.size
            l = (w - 224) // 2; t = (h - 224) // 2
            tiles.append(ip(im.crop((l, t, l + 224, t + 224))))
        inp = torch.stack(tiles).to(device)
        with torch.no_grad():
            v = model.predict_genes(inp, QUERY).cpu().numpy()
        for k, s1 in enumerate(v):
            rows.append((slide, float(s1[S1i].mean()), float(s1[S3i].mean())))
    print(f"  done {slide} ({len(files)} tiles)", flush=True)

import pandas as pd
df = pd.DataFrame(rows, columns=["slide", "S1", "S3"])
print(f"total tiles: {len(df)} across {df['slide'].nunique()} slides", flush=True)

for s in ["S1", "S3"]:
    grand = df[s].mean()
    between = df.groupby("slide")[s].mean().var(ddof=1)
    within = df.groupby("slide")[s].var(ddof=1).mean()
    icc = between / (between + within)
    print(f"{s}: grand_mean={grand:.4f} between_var={between:.5f} within_var={within:.5f} ICC={icc:.3f}", flush=True)

df.to_csv(r"E:\rms_persister\results\tile_level_scores_subset.csv", index=False)
