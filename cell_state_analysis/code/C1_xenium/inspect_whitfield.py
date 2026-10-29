#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect the Whitfield processed h5ad (backed/read-only)."""
import anndata as ad

p = r"E:\rms_persister\data\xenium\processed_data\Spatial_hm_July2025_pub.h5ad"
a = ad.read_h5ad(p, backed="r")
print("shape (cells x genes):", a.shape)
print("obs columns:", list(a.obs.columns))
print("var columns:", list(a.var.columns))
print("obsm keys:", list(a.obsm.keys()))
print("obs head:")
print(a.obs.head(3).to_string())
print()
print("unique cell_type:", sorted(a.obs["cell_type"].unique().tolist()))
print()
print("cell_type counts:")
print(a.obs["cell_type"].value_counts().to_string())
print()
print("cell_class counts:")
print(a.obs["cell_class"].value_counts().to_string())
print()
print("cell_type_2 values:")
print(a.obs["cell_type_2"].value_counts().to_string())
print()
print("cell_labels values (first 20):")
print(a.obs["cell_labels"].value_counts().head(20).to_string())
print()
print("n leiden clusters:", a.obs["leiden"].nunique())
a.file.close()
