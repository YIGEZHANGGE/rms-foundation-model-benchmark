#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from huggingface_hub import hf_hub_download

LOCAL = r"E:\rms_persister\src\C2_zeroshot\DeepSpotM_weights"
for fname in ["config.json", "tokens.csv", "model.safetensors"]:
    try:
        p = hf_hub_download("ratschlab/DeepSpotM", fname, local_dir=LOCAL)
        print(f"[OK] {fname} -> {p}")
    except Exception as e:
        print(f"[ERR] {fname}: {type(e).__name__} {str(e)[:200]}")
