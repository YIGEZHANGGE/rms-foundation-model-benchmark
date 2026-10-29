#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct DeepSpot-M zero-shot prediction on BCH slides (signature genes only).

Clean, bridge-free readout of S1/S3/S4 for the independent BCH cohort. Feeds
O1 (definitive BCH ordering) and I1 (S3 x PAX-FOXO1 fusion) once D3 frees GPU.
"""
import os, glob, time, csv
import numpy as np
import random
import torch
from PIL import Image
from deepspotm import DeepSpotM

WEIGHTS = r"E:\rms_persister\src\C2_zeroshot\DeepSpotM_weights"
SIG = r"E:\rms_persister\results\signatures"
PATCHES = r"F:\Pathology\Pathology_patch_Normalize_quality"
OUT = r"E:\rms_persister\results\D_robustness"
MAX_PATCHES = 1000
BATCH = 64
SEED = 42


def load_genes(fn):
    g = []
    with open(os.path.join(SIG, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            g.append(r["gene"])
    return g


def main():
    S1 = load_genes("S1_progenitor.tsv")
    S3 = load_genes("S3_convergent.tsv")
    S4 = load_genes("S4_ieg_stress.tsv")
    ALL = sorted(set(S1 + S3 + S4))
    model, image_processor = DeepSpotM.from_pretrained(WEIGHTS, source="scgpt")
    model = model.eval()
    device = next(model.parameters()).device
    mg = set(model.gene_names)
    QUERY = [g for g in ALL if g in mg]
    print(f"model on {device}; {len(QUERY)}/{len(ALL)} signature genes queryable", flush=True)

    slides = sorted([d for d in os.listdir(PATCHES) if os.path.isdir(os.path.join(PATCHES, d))])
    print(f"{len(slides)} BCH slides", flush=True)
    out_csv = os.path.join(OUT, "bch_slide_scores.csv")
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide", "S1", "S3", "S4", "n_patches"])
        for i, slide in enumerate(slides):
            sdir = os.path.join(PATCHES, slide)
            files = sorted(glob.glob(os.path.join(sdir, "*.jpg")))
            if len(files) > MAX_PATCHES:
                rng = random.Random(SEED)
                files = rng.sample(files, MAX_PATCHES)
            acc = {"S1": [], "S3": [], "S4": []}
            n = 0
            for b in range(0, len(files), BATCH):
                tiles = []
                for fp in files[b:b + BATCH]:
                    im = Image.open(fp).convert("RGB")
                    wd, ht = im.size
                    l = (wd - 224) // 2; t = (ht - 224) // 2
                    tiles.append(image_processor(im.crop((l, t, l + 224, t + 224))))
                inp = torch.stack(tiles).to(device)
                with torch.no_grad():
                    vals = model.predict_genes(inp, QUERY).cpu().numpy()
                for sig_name, genes in [("S1", S1), ("S3", S3), ("S4", S4)]:
                    idx = [QUERY.index(g) for g in genes if g in QUERY]
                    if idx:
                        acc[sig_name].append(vals[:, idx].mean(axis=1))
                n += len(tiles)
            row = [slide]
            for k in ["S1", "S3", "S4"]:
                row.append(float(np.concatenate(acc[k]).mean()) if acc[k] else float("nan"))
            row.append(n)
            w.writerow(row)
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(slides)}] {slide} done; {time.time()-t0:.0f}s", flush=True)
    print(f"DONE {len(slides)} slides in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
