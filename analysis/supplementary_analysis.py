# HISTORICAL (v1.0). This script reproduces the earlier single-split analysis; its bootstrap resamples individual
# files, not capture groups. The v2 analysis (capture-group bootstrap, matched design) is in v2/analysis/.
"""Supplementary analysis from the reproducibility archive outputs (no images needed).
Usage: python supplementary_analysis.py <archive_dir> <out_dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
A = Path(sys.argv[1]); O = Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
CLASSES = ["freshapples","rottenapples","freshbanana","rottenbanana","freshoranges","rottenoranges"]
pr = pd.read_csv(A/"S3_heldout_predictions.csv"); pairs = pd.read_csv(A/"S2_near_duplicate_pairs.csv")
man = pd.read_csv(A/"S1_manifest_with_split.csv")
yt = pr.label.map(CLASSES.index).to_numpy(); yp = pr.predicted.map(CLASSES.index).to_numpy()
rng = np.random.default_rng(42); B = 10000
accs, f1s = np.empty(B), np.empty(B)
for b in range(B):
    i = rng.integers(0, len(yt), len(yt))
    accs[b] = (yt[i] == yp[i]).mean()
    f1s[b] = precision_recall_fscore_support(yt[i], yp[i], average="macro", zero_division=0)[2]
res = {"n_test": int(len(yt)), "accuracy": float((yt==yp).mean()),
       "accuracy_ci95": [float(np.percentile(accs,2.5)), float(np.percentile(accs,97.5))],
       "macro_f1": float(precision_recall_fscore_support(yt,yp,average="macro",zero_division=0)[2]),
       "macro_f1_ci95": [float(np.percentile(f1s,2.5)), float(np.percentile(f1s,97.5))]}
# fruit-level and condition-level errors
fruit = lambda k: k.replace("fresh","").replace("rotten","")
cond = lambda k: "fresh" if k.startswith("fresh") else "rotten"
wrong = pr[pr.label != pr.predicted]
res["errors_total"] = int(len(wrong))
res["errors_same_fruit_wrong_condition"] = int(((wrong.label.map(fruit)==wrong.predicted.map(fruit))).sum())
res["errors_wrong_fruit_same_condition"] = int(((wrong.label.map(fruit)!=wrong.predicted.map(fruit)) & (wrong.label.map(cond)==wrong.predicted.map(cond))).sum())
res["errors_wrong_fruit_wrong_condition"] = int(((wrong.label.map(fruit)!=wrong.predicted.map(fruit)) & (wrong.label.map(cond)!=wrong.predicted.map(cond))).sum())
res["fresh_vs_rotten_binary_accuracy"] = float((pr.label.map(cond)==pr.predicted.map(cond)).mean())
res["fruit_type_accuracy"] = float((pr.label.map(fruit)==pr.predicted.map(fruit)).mean())
# near-duplicate structure by Hamming distance
dist = []
for d in (0, 2, 4, 6):
    q = pairs[pairs.hamming <= d]
    tw = q[((q.split_a=="test")&(q.split_b=="train"))|((q.split_b=="test")&(q.split_a=="train"))]
    s = set(tw.loc[tw.split_a=="test","file_a"]) | set(tw.loc[tw.split_b=="test","file_b"])
    has = pr.relative_path.isin(s)
    def m(mask):
        if mask.sum()==0: return None
        a = yt[mask.to_numpy()]; b = yp[mask.to_numpy()]
        return {"n": int(mask.sum()), "correct": int((a==b).sum()), "accuracy": float((a==b).mean()),
                "macro_f1": float(precision_recall_fscore_support(a,b,average="macro",zero_division=0)[2])}
    dist.append({"max_hamming": d, "pairs": int(len(q)), "cross_label_pairs": int(q.cross_label.sum()),
                 "test_with_train_twin": m(has), "test_without_train_twin": m(~has)})
res["twin_stratified"] = dist
res["source_split_counts"] = man.groupby(["source_split"]).size().to_dict()
res["image_modes"] = man["mode"].value_counts().to_dict()
res["width_range"] = [int(man.width.min()), int(man.width.max()), float(man.width.median())]
res["height_range"] = [int(man.height.min()), int(man.height.max()), float(man.height.median())]
res["class_counts"] = man.label.value_counts().reindex(CLASSES).to_dict()
res["split_by_class"] = pd.crosstab(man.label, man.split).reindex(CLASSES).to_dict()
# Per-class metrics on the test subset with no exact rotation/flip twin (Hamming 0) in training
q = pairs[pairs.hamming == 0]
tw = q[((q.split_a=="test")&(q.split_b=="train"))|((q.split_b=="test")&(q.split_a=="train"))]
s = set(tw.loc[tw.split_a=="test","file_a"]) | set(tw.loc[tw.split_b=="test","file_b"])
keep = ~pr.relative_path.isin(s).to_numpy()
p_, r_, f_, n_ = precision_recall_fscore_support(yt[keep], yp[keep], labels=range(6), zero_division=0)
pd.DataFrame({"Class": CLASSES, "n": n_, "Precision": (p_*100).round(2), "Recall": (r_*100).round(2), "F1": (f_*100).round(2)}).to_csv(O/"metrics_test_without_train_twin.csv", index=False)
(O/"supplementary_analysis.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
