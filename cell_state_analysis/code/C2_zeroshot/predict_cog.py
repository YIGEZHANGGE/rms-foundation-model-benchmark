#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSpot-M zero-shot prediction of signature genes on COG slides (batched)."""
import glob
import os
import time
import csv
import numpy as np
import random
import torch
from PIL import Image
from deepspotm import DeepSpotM

WEIGHTS = r"E:\rms_persister\src\C2_zeroshot\DeepSpotM_weights"
SIG = r"E:\rms_persister\results\signatures"
PATCHES = r"F:\Pathology\IDC_RMS_patches"
OUT = r"E:\rms_persister\results\cog_pred"
os.makedirs(OUT, exist_ok=True)
MAX_PATCHES = 1000


def load_genes(fn):
    g = []
    with open(os.path.join(SIG, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            g.append(r["gene"])
    return g


S1 = load_genes("S1_progenitor.tsv")
S3 = load_genes("S3_convergent.tsv")
S4 = load_genes("S4_ieg_stress.tsv")
ALL_GENES = sorted(set(S1 + S3 + S4))

model, image_processor = DeepSpotM.from_pretrained(WEIGHTS, source="scgpt")
model = model.eval()
device = next(model.parameters()).device
model_genes = set(model.gene_names)
QUERY = [g for g in ALL_GENES if g in model_genes]
print(f"model on {device}; {len(QUERY)}/{len(ALL_GENES)} signature genes queryable", flush=True)


def predict_slide(slide_dir, batch=64):
    files = sorted(glob.glob(os.path.join(slide_dir, "*.jpg")))
    if len(files) > MAX_PATCHES:
        rng = random.Random(42)
        files = rng.sample(files, MAX_PATCHES)
    scores = {g: [] for g in ["S1", "S3", "S4"]}
    n_patches = 0
    for i in range(0, len(files), batch):
        tiles = []
        for f in files[i:i + batch]:
            im = Image.open(f).convert("RGB")
            w, h = im.size
            l = (w - 224) // 2; t = (h - 224) // 2
            tiles.append(image_processor(im.crop((l, t, l + 224, t + 224))))
        inp = torch.stack(tiles).to(device)
        with torch.no_grad():
            vals = model.predict_genes(inp, QUERY)  # (batch, n_query)
        vals = vals.cpu().numpy()
        # patch-level signature score = mean log1p-CPM of signature genes (no cross-patch z)
        for sig_name, genes in [("S1", S1), ("S3", S3), ("S4", S4)]:
            idx = [QUERY.index(g) for g in genes if g in QUERY]
            if not idx:
                continue
            sub = vals[:, idx]
            scores[sig_name].append(sub.mean(axis=1))
        n_patches += len(files[i:i + batch])
    return {k: np.concatenate(v) if v else np.array([]) for k, v in scores.items()}, n_patches


if __name__ == "__main__":
    import sys
    # batch mode: all slides -> per-slide score CSV
    slides = sorted(os.listdir(PATCHES))
    print(f"batch: {len(slides)} slides found", flush=True)
    out_csv = os.path.join(OUT, "cog_slide_scores.csv")
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide", "S1", "S3", "S4", "n_patches"])
        for i, slide in enumerate(slides):
            sdir = os.path.join(PATCHES, slide)
            if not os.path.isdir(sdir):
                continue
            print(f"processing {slide}...", flush=True)
            scores, n = predict_slide(sdir)
            w.writerow([slide, float(scores["S1"].mean()), float(scores["S3"].mean()),
                        float(scores["S4"].mean()), n])
            f.flush()
            print(f"  done {slide}: S1={scores['S1'].mean():.3f}", flush=True)
            if (i + 1) % 5 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(slides)}] {el:.0f}s elapsed (slide={slide})", flush=True)
    print(f"done {len(slides)} slides in {time.time()-t0:.0f}s", flush=True)
