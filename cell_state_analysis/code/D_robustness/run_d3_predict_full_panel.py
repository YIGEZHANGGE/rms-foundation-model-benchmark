#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D3 — DeepSpot-M full-panel (19,338-gene) zero-shot prediction on COG slides.

Produces a slide x gene matrix of per-slide mean predicted expression so that
D3 permutation can sample random signatures from the SAME predicted gene space.
Gene decoding is chunked (2,000 genes at a time) to avoid OOM in the
cross-attention gene decoder; patch encoding is done once per batch.
"""
import os, glob, time, json
import numpy as np
import torch
from PIL import Image
from deepspotm import DeepSpotM

WEIGHTS = r"E:\rms_persister\src\C2_zeroshot\DeepSpotM_weights"
PATCHES = r"F:\Pathology\IDC_RMS_patches"
COHORT = r"C:\Users\onekey\Documents\Codex\2026-08-02\read-project-context-md-in-this\outputs\data_snapshot\data\cohort_and_folds_public.json"
OUT = r"E:\rms_persister\results\D_robustness"
os.makedirs(OUT, exist_ok=True)

MAX_PATCHES = 1000
BATCH = 64
GENE_CHUNK = 2000
SEED = 42


def main():
    cohort_ids = set(json.load(open(COHORT, encoding="utf-8"))["labels"].keys())
    model, image_processor = DeepSpotM.from_pretrained(WEIGHTS, source="scgpt")
    model = model.eval()
    device = next(model.parameters()).device
    gene_names = list(model.gene_names)
    n_genes = len(gene_names)
    all_idx = torch.arange(n_genes)
    print(f"model on {device}; n_genes={n_genes}", flush=True)

    slides = sorted(os.listdir(PATCHES))
    slides = [s for s in slides if s in cohort_ids]
    print(f"{len(slides)} in-cohort slides to predict", flush=True)

    rows = []  # (slide, gene-mean vector)
    t0 = time.time()
    for i, slide in enumerate(slides):
        sdir = os.path.join(PATCHES, slide)
        files = sorted(glob.glob(os.path.join(sdir, "*.jpg")))
        if len(files) > MAX_PATCHES:
            rng = np.random.RandomState(SEED)
            files = [files[j] for j in rng.choice(len(files), MAX_PATCHES, replace=False)]
        gene_sum = np.zeros(n_genes, dtype=np.float64)
        n_patches = 0
        for b in range(0, len(files), BATCH):
            tiles = []
            for f in files[b:b + BATCH]:
                im = Image.open(f).convert("RGB")
                w, h = im.size
                l = (w - 224) // 2; t = (h - 224) // 2
                tiles.append(image_processor(im.crop((l, t, l + 224, t + 224))))
            inp = torch.stack(tiles).to(device)
            with torch.no_grad():
                patch_tokens = model.backbone.forward_patch_tokens(inp)
                for g0 in range(0, n_genes, GENE_CHUNK):
                    gi = all_idx[g0:g0 + GENE_CHUNK].to(device)
                    expr, _, _ = model.gene_decoder(patch_tokens, need_weights=False, gene_indices=gi)
                    gene_sum[g0:g0 + GENE_CHUNK] += expr.cpu().numpy().sum(axis=0)
            n_patches += len(tiles)
        gene_mean = gene_sum / max(n_patches, 1)
        rows.append((slide, gene_mean))
        if (i + 1) % 10 == 0 or i == len(slides) - 1:
            el = time.time() - t0
            print(f"[{i+1}/{len(slides)}] {slide} done; {el:.0f}s elapsed", flush=True)

    slides_out = [r[0] for r in rows]
    mat = np.vstack([r[1] for r in rows]).astype(np.float32)
    np.save(os.path.join(OUT, "D3_slide_gene_mean.npy"), mat)
    with open(os.path.join(OUT, "D3_slides.txt"), "w") as f:
        f.write("\n".join(slides_out))
    with open(os.path.join(OUT, "D3_genes.txt"), "w") as f:
        f.write("\n".join(gene_names))
    print(f"DONE {len(slides_out)} slides x {n_genes} genes in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
