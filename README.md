# RMS Foundation-Model Benchmark (v3.22b)

Companion code and shareable data for: **Computational Pathology for Rhabdomyosarcoma Histological Subtyping: A Multi-Model Evaluation with Cross-Center Validation**.

## What is here

- `code/` — Python scripts for feature handling, MIL training, five-fold cross-validation, external validation, statistics, survival analysis, and attention analysis. The custom MAMMOTH classifier is defined in `code/train_diagnosis_v3.py`.
- `data/cohort_and_folds_public.json` — the frozen COG cohort (n = 390), labels, and the five shared stratified folds.
- `data/predictions/diagnosis_oof_*.csv` — out-of-fold predictions for the COG training cohort (five models).
- `data/results/*.json` and `*.csv` — aggregate (de-identified, non-patient-level) results used to reproduce every table and figure.

## Data policy

Only COG-derived, de-identified, non-patient-level data are shared here. The
BCH per-patient prediction files (`bch_predictions_v3_*.csv`) and all raw
whole-slide images and per-patient clinical data are withheld under the COG
and BCH data-use agreements and institutional privacy rules (see the
manuscript Data availability statement).

## Requirements

See `requirements.txt`. Feature extraction requires the frozen model checkpoints (CONCH, Phikon-v2, H-optimus-0, Prov-GigaPath, Virchow2); training and evaluation use PyTorch and the aggregate data above.

## Reproducibility

All random seeds are fixed (data split and bootstrap: seed 42; per-fold training seeds are deterministic and model/fold-specific). Each result JSON can be reproduced by running the corresponding script in `code/` against the shared aggregate data.

## License

Research-use license (see `LICENSE`).
