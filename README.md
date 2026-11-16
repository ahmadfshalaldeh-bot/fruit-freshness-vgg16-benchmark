# One Photograph, Nine Files

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24273731.svg)](https://doi.org/10.5281/zenodo.24273731)

Code, provenance records, split files and results for the article submitted to *Applied Sciences* (MDPI):
**Shalaldeh, A. "One Photograph, Nine Files: Provenance Audit and Sibling-Controlled Evaluation of a Public Fresh and Rotten Fruit Image Dataset."**

Version 2.0 ([10.5281/zenodo.24273731](https://doi.org/10.5281/zenodo.24273731)) is the version used in the article. The concept DOI [10.5281/zenodo.24242046](https://doi.org/10.5281/zenodo.24242046) always resolves to the latest version. Version 1.0 ([10.5281/zenodo.24242047](https://doi.org/10.5281/zenodo.24242047)) holds the earlier single-split configuration and is kept for reference.

## What the audit found

The Kaggle collection [*Fruits Fresh and Rotten for Classification*](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) (fresh and rotten apple, banana and orange) downloads as 27,198 files:

* every file appears twice (`dataset/` and `dataset/dataset/`), leaving **13,599 unique images**;
* these are **nine stored versions of 1,511 screen captures**: the capture, rotations of 15–75°, a salt-and-pepper copy, a translated copy and a vertical flip (the operation is the file-name prefix);
* **all 2,698 images in the collection's own test folder have a sibling in its training folder**;
* 29 zero-distance perceptual-hash links join 13 pairs of families; merging them gives **1,498 capture groups**;
* a perceptual hash recovers at most 8.6% of the 54,396 sibling pairs at the thresholds tested; VGG16 embeddings rank siblings with an AUC of 0.876.

## Matched contamination study (E3)

Both arms share a donor-depleted common training base. The clean arm adds seven derived versions of a same-class donor family for each test family; the exposed arm adds the seven corresponding versions of the test family itself. Training size, class counts, version mix, initial weights and the number of optimiser updates are identical; both arms are evaluated on the same held-out original captures and withheld derived versions.

| Backbone (10 paired fits each) | Clean | Exposed | Difference | Errors (clean → exposed) |
|---|---|---|---|---|
| Frozen VGG16, original captures | 98.55% | 99.37% | +0.83 pp | 44 → 19 |
| Fine-tuned VGG16, original captures | 97.95% | 99.97% | +2.02 pp | 62 → 1 |

## Repository layout

```
v2/
  notebook/fruit_v2_experiments.ipynb   Colab notebook that ran every v2 experiment (E1-E5)
  src/pipeline_v2.py, design_v2.py      the same code as plain Python; design_v2.py defines folds, donors and statistics
  analysis/                             read-only analysis: regenerates numbers, tables and figures from v2/results
  results/
    audit/      file manifest with checksums, family and capture-group identifiers, hash pairs, cross-family links, corner checks
    embed/      embedding-detection summary (E2)
    designs/    role of every file in every E3 split (base, donor bundle, exposure bundle, validation, evaluation, unused)
    runs/       per-fit run records, training histories and predictions with class probabilities; initial-weight hashes
    environment_v2.json, requirements_lock_v2.txt
    SHA256SUMS.txt
results/, src/, analysis/, notebooks/, figures/   version 1.0 materials (historical)
```

## Reproducing the reported numbers without retraining

```bash
pip install numpy pandas scipy scikit-learn matplotlib
cd v2/analysis
python build_v2.py ../../results/image_split ../../results/photo_split ../results ./out
python supplement_v2.py ../results ./out/supp ./out     # optional: Supplementary PDF (needs pdflatex)
```

`out/numbers_used_v2.json` holds every number reported in the article; `out/TableS1_E3_pairs.csv` and `out/TableS2_split_composition.csv` are Supplementary Tables S1 and S2; `out/Figures/` holds Figures 1–3. No GPU and no images are needed.

| Article item | Source records | Produced by |
|---|---|---|
| Tables 3 and 4, Section 4.1–4.2 | `v2/results/audit/*`, `v2/results/embed/*`, v1.0 `results/*/S1_*`, `S2_*` | `build_v2.py` |
| Table 5, Figure 3, Tables 6–7, Table S1 | `v2/results/runs/E3*` | `build_v2.py`, `extra_v2.py`, `figures_v2.py` |
| Table 8 | `v2/results/runs/E4*`, `E5*`; v1.0 `results/*/S3_heldout_predictions.csv` | `build_v2.py` |
| Table S2 | `v2/results/designs/*` | `extra_v2.py` |
| Table S3 | `v2/results/audit/cross_family_tau0_pairs.csv` | `supplement_v2.py` |

## Re-running the experiments

Open `v2/notebook/fruit_v2_experiments.ipynb` in Google Colab (A100 GPU, High RAM) and run all cells. The notebook downloads the dataset with `kagglehub`, writes every result to Google Drive after each fit and skips finished fits on restart. The initial-weight files (`*.init_gpuaug.weights.h5`) are not in the repository because of their size; their SHA-256 digests are in `v2/results/runs/init_weight_hashes.csv` and the files are available from the author on request.

## Notes on version 1.0 files

* `results/photo_families.csv` now carries the photo-level split (an earlier commit carried the image-level split by mistake).
* `analysis/build_paper.py`, `make_figures.py` and `supplementary_analysis.py` reproduce the v1.0 analysis; the v1.0 bootstrap resamples individual files, not capture groups.
* The `SHA256SUMS.txt` files in `results/image_split` and `results/photo_split` also list the v1.0 model weight files, which are not in the repository because of their size.

## Licence and data

Code and derived records: MIT. The images belong to the Kaggle dataset and are not redistributed; its licence is listed as unknown.
