# One Photograph, Nine Images: provenance-aware benchmarking of a frozen VGG16 fresh/rotten fruit classifier

Code, split assignments and results for the article submitted to *Applied Sciences* (MDPI):
**Shalaldeh, A. "One Photograph, Nine Images: Provenance-Aware Benchmarking of a Frozen VGG16 Classifier for Fresh and Rotten Fruit."**

## Key finding

The Kaggle collection [*Fruits Fresh and Rotten for Classification*](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
(6 classes: fresh/rotten apple, banana, orange) downloads as **27,198 files**:

* every file appears twice (`dataset/` and `dataset/dataset/`), leaving **13,599 unique images**;
* those 13,599 images come from only **1,511 original photographs**, each stored in **9 versions**:
  original, rotated 15°/30°/45°/60°/75°, salt-and-pepper, translation, vertical flip
  (the operation is the file-name prefix, e.g. `rotated_by_15_Screen Shot …png`);
* perceptual hashing recovers at most **8.6 %** of the 54,396 sibling pairs, so hash-based de-duplication is not enough.

With a conventional random image-level split, **every test image has a sibling of the same photograph in training**.

| Protocol (frozen VGG16, identical pipeline, seed 42) | Test images | Accuracy (95 % CI, photo-clustered bootstrap) | Macro F1 |
|---|---|---|---|
| Image-level split (stratified, 70/15/15) | 2,040 | 97.01 % (95.9–97.9) | 96.94 % |
| **Photo-level split** (all 9 versions of a photo in one partition) | 2,052 | **94.88 % (92.6–96.9)** | **94.69 %** |

The error rate rises from 2.99 % to 5.12 % (≈ +71 %), and F1 is lower for all six classes.
**Please use the photo-level split** (`results/photo_split/S1_manifest_with_split.csv`, column `split`) when benchmarking on this collection.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/fruit_vgg16_reproducibility.ipynb` | Colab notebook (A100 GPU recommended). `SPLIT_MODE = "family"` (photo-level) or `"image"` |
| `src/pipeline.py` | Same code as a Python script |
| `src/pipeline_v1_image_split.py` | Exact pipeline version that produced `results/image_split` (run with `PER_CLASS = None`) |
| `results/photo_families.csv` | Photograph identifier (`family`) and stored version (`op`) for every image |
| `results/image_split/`, `results/photo_split/` | For each arm: file inventory with SHA-256 (S1a), manifest with split (S1), perceptual-hash audit (S2), held-out predictions with class probabilities (S3), confusion matrix, class metrics, training history, model summary, environment record and `pip freeze` lock file, `SHA256SUMS.txt` |
| `results/numbers_used.json` | Every number quoted in the manuscript, computed from the two result folders |
| `analysis/` | Scripts that compute the manuscript numbers and draw the figures from the result folders (no images needed) |
| `figures/` | Manuscript figures (no dataset photographs are reproduced) |

## Reproduce

1. Open `notebooks/fruit_vgg16_reproducibility.ipynb` in Google Colab, choose a GPU runtime, and **Run all**.
   The notebook downloads the dataset with `kagglehub`, audits it, splits it, trains, evaluates and writes a reproducibility archive.
2. Re-draw the figures from the archived results:
   `python analysis/make_figures.py results/image_split results/photo_split figures/`

Environment of the reported runs: NVIDIA A100-SXM4-80GB (Google Colab), Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2, scikit-learn 1.6.1.
Trained weights (~60 MB per arm) are available from the author on request.

## Data licence note

The images are **not** redistributed here; the Kaggle page lists the dataset licence as unknown. The manifests list relative paths and SHA-256 checksums so the exact analysed set can be rebuilt from a fresh download.

## Licence and citation

Code: MIT (see `LICENSE`). Please cite the article (see `CITATION.cff`).
