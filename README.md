# RMS foundation-model benchmark — analysis code and shareable data

Companion repository for the manuscript:

> **Where routine H&E can and cannot guide rhabdomyosarcoma subtyping: a cross-center evaluation of
> pathology foundation models.**

This repository holds the code and the de-identified, aggregate data needed to check the reported
numbers. Figure-drawing scripts, manuscript files and anything that identifies a patient are not
included.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | Training, evaluation, statistics and analysis scripts |
| `data/cohort_and_folds_public.json` | Frozen COG cohort (n = 390), labels and the five shared folds |
| `data/predictions/` | Out-of-fold COG predictions, one file per encoder |
| `data/results/` | Aggregate results for every analysis reported in the paper |
| `cell_state_analysis/` | Self-contained sub-repository: signatures, pipeline and de-identified slide-level scores |

## Scripts

| Script | Purpose |
| --- | --- |
| `train_diagnosis_v3.py` | Five-fold training of the five encoders (MAMMOTH classifier) |
| `aggregate_v3.py` | Internal metrics, per-fold tables, confusion matrices |
| `external_validation_v3.py` | Application of the frozen models to the external cohort |
| `architecture_ablation_v3.py`, `architecture_ablation_v3_run2.py`, `ablation_run_compare_v3.py` | Pooling-architecture ablation and run-to-run stability |
| `sensitivity_v3.py` | Adult-exclusion and related sensitivity analyses |
| `bch_histology_fusion_v3.py` | Histology-by-fusion cross-tabulation |
| `table1_v3.py` | Cohort description |
| `foxo1_v3_dedup.py` | PAX-FOXO1 classifier on 218 de-duplicated patients |
| `survival_v4.py` (uses helpers from `survival_v3.py`) | Event-free survival on adjudicated event status |
| `tissue_content_v3.py` | Patch-level tissue-content audit |
| `spindle_binary_v3.py`, `spindle_binary_external_v3.py` | Two-class embryonal-versus-spindle sensitivity analysis |
| `check_mca_folds.py` | Verification that fold-identical confusion matrices are not duplicated arrays |

## What is not here

* Raw whole-slide images, patches and COG clinical tables — governed by the data-use agreements.
* Per-patient BCH predictions, per-patient PAX-FOXO1 probabilities and per-patient tissue metrics —
  withheld for the same reason. Aggregate outputs for every analysis that uses them are in
  `data/results/`, so each reported number can be checked against a stored file.
* Figure-generation code and manuscript files.

## Reproducibility notes

* Seeds: data split and bootstrap 42; per-fold training seeds are deterministic and model/fold-specific.
* Internal estimates are single-model out-of-fold predictions; external probabilities are averaged over
  the five fold models. The two are not constructed identically, and the manuscript says so.
* Two analyses were re-run before submission and their numbers changed: PAX-FOXO1 (219 records with
  one patient duplicated → 218 unique patients) and survival (event status imputed from blanks →
  adjudicated status, n = 258). Only the current outputs are kept; `CHANGELOG.md` records the change.

## License

Research-use licence; see [LICENSE](LICENSE).
