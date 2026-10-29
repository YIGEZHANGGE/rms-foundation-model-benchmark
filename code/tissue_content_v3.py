# -*- coding: utf-8 -*-
"""
Tissue-content audit (manuscript: Methods "Histology and sampling"; Figure 4c;
Results "Histology and tissue content").

Two steps, both included:

  1. extract_patch_metrics()  - reads 512-px patch JPEGs of one slide and returns
     nuclear density and white-space fraction. Requires patch access (imaging data
     are governed by the COG/BCH data-use agreements and are not distributed).

  2. summarise()              - takes per-patient metrics plus specimen type and
     correctness, and reproduces every number reported in the paper.

Definitions (as stated in the Methods)
  * 25 randomly sampled patches per slide, random seed 0
  * nuclear density = mean haematoxylin optical density (colour deconvolution)
  * white space     = fraction of pixels with mean RGB intensity > 0.85
  * patient value   = median across sampled patches
  * comparisons     = two-sided Mann-Whitney tests
"""
import json
import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu

PATCH = 512
N_PATCHES_PER_SLIDE = 25
WHITE_THRESHOLD = 0.85
SEED = 0


def _haematoxylin_od(rgb):
    """Ruzin-style colour deconvolution: OD vector for haematoxylin is (0.65, 0.70, 0.29)."""
    od = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)
    hec = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
    stains = np.linalg.pinv(hec) @ od.reshape(-1, 3).T
    return stains[0].reshape(rgb.shape[:2])


def extract_patch_metrics(patch_paths, n_patches=N_PATCHES_PER_SLIDE, seed=SEED):
    rng = np.random.RandomState(seed)
    if len(patch_paths) > n_patches:
        idx = rng.choice(len(patch_paths), n_patches, replace=False)
        patch_paths = [patch_paths[i] for i in idx]
    dens, white = [], []
    for p in patch_paths:
        im = Image.open(p).convert("RGB")
        if im.size != (PATCH, PATCH):
            im = im.resize((PATCH, PATCH), Image.BILINEAR)
        rgb = np.asarray(im)
        dens.append(float(_haematoxylin_od(rgb).clip(0, None).mean()))
        white.append(float((rgb.mean(axis=2) > WHITE_THRESHOLD).mean()))
    return {"nuclear_density": float(np.median(dens)),
            "white_space": float(np.median(white)),
            "n_patches": len(dens)}


def summarise(metrics_csv):
    """metrics_csv columns: patient_id, nuclear_density, white_space, specimen (0=puncture, 1=biopsy),
    true, pred. Writes data/results/tissue_content_summary_v3.json."""
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    df["correct"] = (df["true"] == df["pred"]).astype(int)
    punc, bio = df[df.specimen == 0], df[df.specimen == 1]
    cor, inc = df[df.correct == 1], df[df.correct == 0]
    out = {
        "n_patients": int(len(df)),
        "by_specimen": {
            "puncture": {"n": int(len(punc)),
                         "median_nuclear_density": round(float(punc.nuclear_density.median()), 3),
                         "median_white_space": round(float(punc.white_space.median()), 2)},
            "biopsy": {"n": int(len(bio)),
                       "median_nuclear_density": round(float(bio.nuclear_density.median()), 3),
                       "median_white_space": round(float(bio.white_space.median()), 2)}},
        "by_correctness_CONCH": {
            "correct": {"n": int(len(cor)),
                        "median_nuclear_density": round(float(cor.nuclear_density.median()), 3),
                        "median_white_space": round(float(cor.white_space.median()), 2)},
            "incorrect": {"n": int(len(inc)),
                          "median_nuclear_density": round(float(inc.nuclear_density.median()), 3),
                          "median_white_space": round(float(inc.white_space.median()), 2)}},
        "mann_whitney": {
            "nuclear_density_p": round(float(mannwhitneyu(cor.nuclear_density, inc.nuclear_density).pvalue), 3),
            "white_space_p": round(float(mannwhitneyu(cor.white_space, inc.white_space).pvalue), 3)},
        "spindle_subgroup": {"n": int((df.true == 2).sum()),
                             "median_nuclear_density": round(float(df[df.true == 2].nuclear_density.median()), 3)},
        "definitions": ("25 random 512-px patches per slide (seed 0); nuclear density = mean haematoxylin optical "
                        "density; white space = fraction of pixels with mean RGB > 0.85; patient value = median across "
                        "patches; specimen 0 = puncture, 1 = biopsy."),
    }
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(summarise(sys.argv[1]), indent=2))
    else:
        print(__doc__)
