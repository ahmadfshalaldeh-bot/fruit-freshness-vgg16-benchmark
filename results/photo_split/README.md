# Reproducibility archive — VGG16 six-class fresh/rotten fruit benchmark

Generated 2026-11-13T14:18:44+00:00. Dataset: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification (downloaded 2026-11-13T14:02:15+00:00).
Source images are NOT redistributed; the manifest lists each file's relative path and SHA-256 so the exact set can be rebuilt.

| File | Manuscript item |
|---|---|
| S1_manifest_with_split.csv | Supplementary Data S1 — file manifest, labels, dimensions, checksums, split |
| S1a_raw_inventory.csv | Full inventory of the download, before de-duplication/sampling |
| S2_duplicate_audit_summary.json, S2_near_duplicate_pairs.csv | Supplementary Data S2 — duplicate / near-duplicate audit |
| S3_heldout_predictions.csv | Supplementary Data S3 — held-out predictions and class probabilities |
| Code_S1_pipeline.py (or the .ipynb) | Supplementary Code S1 |
| S_Code_S2_environment.json, S_Code_S2_requirements_lock.txt | Supplementary Code S2 — environment |
| S_Figure_S1_model_summary.txt | Supplementary Figure S1 — model summary |
| Table_3_class_metrics.csv, confusion_matrix_counts.csv, Figure_3_* | Table 3 and Figure 3 |
| training_history.csv, Figure_S2_training_curves.png | Logged training history |
| vgg16_best.weights.h5 | Trained weights (if included) |
| run_record.json | Configuration, counts, and headline results |

Held-out result: 1947/2052 = 94.88% accuracy; macro F1 = 94.69%.
To reproduce: run Code_S1 with the same configuration (seed 42).
