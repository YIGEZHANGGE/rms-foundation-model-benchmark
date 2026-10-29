# Changelog

## 28 October 2026 — aggregate result files added

Added the two aggregate files that back analyses reported after the initial release: `data/results/deferral_summary_v3.json`
(Table S9) and `data/results/blinded_review_summary_v3.json` (Table S4). No analysis code changed;
`data/results/` now holds 25 files.

## 28 October 2026 — result files reduced to those cited

`data/results/` reduced to the files that back a number in the manuscript. Removed: `km_logrank_v3.json`
(no Kaplan-Meier analysis is reported), `foxo1_dca_v3.json` (decision curves are reported for the
alveolar output only) and `pairwise_tests_v3_fixed.json` (superseded by `internal_pairwise_v3.json`).
23 result files remain, one per reported analysis.

## 28 October 2026 — repository trimmed to code and data

Repository trimmed to analysis code and data.

* Removed `code/figures/` (figure-drawing scripts) and all manuscript-related documents.
* Removed `blinded_review/`; the review protocol and agreement tables are part of the manuscript's
  supplementary material rather than the code repository.
* Removed `REPRODUCIBILITY.md`; the script map now sits in `README.md`.
* Removed the superseded outputs `survival_results_v3.json` and `foxo1_results_v3.json`, and the
  superseded `foxo1_v3.py`. Only the results used in the paper are kept.

## 28 October 2026 — two analyses re-run

Two analyses were re-run against the manuscript and their numbers changed:

* **PAX-FOXO1** (`foxo1_v3_dedup.py`): the earlier run cross-validated over 219 records with patient
  1002311 duplicated, hence present in two folds. Repeated on 218 unique patients. Pooled AUROC
  0.740–0.807 (was 0.741–0.770); within-ARMS AUROC 0.346–0.519 (was 0.390–0.440).
* **Survival** (`survival_v4.py`): event status had been imputed from blank fields. It is now taken
  from the adjudicated field: 258 patients, 75 events, scaler and PCA fitted inside each fold.
  C-index 0.468 (morphology), 0.544 (clinical), 0.630 (combined).

Also added: `tissue_content_v3.py`, `spindle_binary_v3.py`, `spindle_binary_external_v3.py`,
`check_mca_folds.py`, and the corresponding aggregate outputs.

## earlier snapshot

Initial public release: five-encoder training, five-fold cross-validation, external validation and
the aggregate results of that stage. Superseded by a later revision.
