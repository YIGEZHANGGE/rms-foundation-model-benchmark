# Cell-state-resolved morphological analysis

This directory reproduces the exploratory cell-state analysis in the
manuscript (Results: "Secondary analyses", cell-state paragraph; Supplementary Methods;
Supplementary Fig. S10; Supplementary Tables S17-S19). It derives developmental cell-state
signatures from public single-cell rhabdomyosarcoma atlases, scores them in an
independent expression cohort, projects them onto H&E slides with a zero-shot
transcriptome predictor, and tests their fusion and survival associations.

## Directory layout

- `signatures/` — final gene lists for S1 (progenitor), S3 (convergent),
  S4 (immediate-early gene negative control), and S5 (1,000 size-matched
  random gene sets).
- `code/` — pipeline scripts ordered by phase.
- `results/` — de-identified slide-level scores and summary tables. COG slide
  IDs are `RMS####`; BCH slide IDs are `1002###`. No protected health
  information, raw WSIs, patches, or patient-level clinical tables are
  included.

## Public inputs

- Danielli et al., Nature Communications 2024, single-cell RMS atlas,
  Supplementary Data 2 (used to build S1).
- E-TABM-1202 (ArrayExpress, HG-U133 Plus 2.0), fusion-negative RMS with
  survival annotation; the endpoint is death from disease.
- Whitfield et al. Xenium data (Zenodo), convergent high-risk state (S3).
- DeepSpot-M (github.com/ratschlab/DeepSpotM), zero-shot transcriptome
  predictor from H&E; cite the preprint and honor its license.

## Reproduction order

1. `code/A_signature/` — download the Danielli marker table, extract
   per-state genes, build S4/S5 controls, and check signature independence.
2. `code/B_target/` — map E-TABM-1202 probes to genes, score S1-S4, and run
   the survival gate (`run_gateA.py`).
3. `code/C1_xenium/` — reconstruct S3 from the Xenium malignant-cell
   convergent state.
4. `code/C2_zeroshot/` — predict signature genes on COG slides with DeepSpot-M
   (`predict_cog.py`).
5. `code/C3_cog/` — COG survival analysis and tile-level variance
   decomposition.
6. `code/D_robustness/` — full-panel prediction for the permutation null,
   permutation tests, encoder-heterogeneity bridges, BCH direct readout,
   fusion-axis tests, and attention-to-cell-state associations.

## Results files

| File | Content |
|---|---|
| `results/cog_slide_scores_with_histology.csv` | COG slide-level S1/S3/S4 scores with histology and patch count |
| `results/bch_slide_scores.csv` | BCH slide-level S1/S3/S4 scores and patch count |
| `results/D3_permutation.csv` | observed HR versus 1,000 random-signature nulls |
| `results/D3_permutation_null.npz` | raw null distributions used for Supplementary Fig. S10c |
| `results/D4_encoder_heterogeneity.csv` | encoder-specific bridge R² and survival associations |
| `results/T1T3_fusion_axis.csv` | COG histology and BCH fusion-axis comparisons |
| `results/I2_attention_cellstate.csv` | attention metrics versus S1/S3/S4 scores |
| `results/O1_bch_ordering.csv` | encoder-specific BCH histology ordering |
| `results/D3_genes.txt` | DeepSpot-M predicted gene space used to sample random signatures |
| `results/D3_slides.txt` | slide order matching `D3_slide_gene_mean.npy` |

The full slide × gene matrix (`D3_slide_gene_mean.npy`) is not included because
of size; `code/D_robustness/run_d3_predict_full_panel.py` regenerates it from
the patch directory and DeepSpot-M weights.

## Statistical conventions

- Random seed 42 throughout.
- Slide-level scoring uses up to 1,000 patches per slide.
- Cox models use `score + age + stage`; scores are z-scored within cohort; HR
  is reported per 1 SD.
- Event indicators are coded 1 = event and 0 = censored/no event; follow-up
  times are days converted to years.
- Benjamini-Hochberg correction is applied where stated and capped at 1.0.

Coding note: the public COG table does not include a formal data dictionary
that confirms blank event fields mean censored. Treating a missing event
indicator as censored is therefore a documented assumption in the pipeline,
not a source-derived definition. It should be rechecked against the original
COG or Milewski supplementary data before reusing these scripts for a new
analysis.

## Running the scripts

The scripts were developed on a Windows machine with absolute local paths for
patch directories, feature caches, and model weights. Replace the `PATCHES`,
`WEIGHTS`, `CLIN`, and `OUT` variables at the top of each script with the
corresponding paths in your environment. Python dependencies are listed in
`requirements.txt`; the DeepSpot-M package must be installed from its
repository.

## Privacy and licensing

- COG and BCH raw images and patient-level clinical data remain governed by
  their respective data-use agreements and are not distributed here.
- Signature gene lists are derived from public resources and retain their
  original attribution.
- Code in this directory is provided for research reproducibility only and
  does not grant permission to redistribute the underlying model weights or
  clinical data.
