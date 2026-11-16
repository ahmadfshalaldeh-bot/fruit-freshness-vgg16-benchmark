"""HISTORICAL (v1.0): builds the numbers of the earlier single-split configuration. The v2 analysis is in v2/analysis/.
Compute every number used in the manuscript from the two reproducibility archives,
draw the figures, and fill the LaTeX template.

Usage: python build_paper.py <image_split_archive_dir> <family_split_archive_dir> <paper_dir>
"""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

IMG, FAM, PAPER = (Path(a) for a in sys.argv[1:4])
CLASSES = ["freshapples", "rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges"]
NAMES = ["Fresh apple", "Rotten apple", "Fresh banana", "Rotten banana", "Fresh orange", "Rotten orange"]
OP_RE = r"^(rotated_by_\d+|saltandpepper|translation|vertical_flip)_"
VARIANTS = ["original", "rotated_by_15", "rotated_by_30", "rotated_by_45", "rotated_by_60", "rotated_by_75",
            "saltandpepper", "translation", "vertical_flip"]
VNAMES = {"original": "Original", "rotated_by_15": "Rotated 15°", "rotated_by_30": "Rotated 30°",
          "rotated_by_45": "Rotated 45°", "rotated_by_60": "Rotated 60°", "rotated_by_75": "Rotated 75°",
          "saltandpepper": "Salt-and-pepper", "translation": "Translation", "vertical_flip": "Vertical flip"}

def fam_cols(df):
    fn = df["relative_path"].str.split("/").str[-1]
    df = df.copy()
    df["variant"] = fn.str.extract(OP_RE)[0].fillna("original")
    df["family"] = df["label"] + "|" + fn.str.replace(OP_RE, "", regex=True)
    return df

def analyse(arch):
    rr = json.loads((arch / "run_record.json").read_text())
    man = fam_cols(pd.read_csv(arch / "S1_manifest_with_split.csv"))
    pr = fam_cols(pd.read_csv(arch / "S3_heldout_predictions.csv"))
    yt = pr.label.map(CLASSES.index).to_numpy(); yp = pr.predicted.map(CLASSES.index).to_numpy()
    p, r, f, n = precision_recall_fscore_support(yt, yp, labels=range(6), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    o = (pr.variant == "original").to_numpy()
    po, ro_, fo, no = precision_recall_fscore_support(yt[o], yp[o], labels=range(6), zero_division=0)
    _, _, mfo, _ = precision_recall_fscore_support(yt[o], yp[o], average="macro", zero_division=0)
    # photo-clustered bootstrap (resample photographs, keep all their stored versions)
    fams, idx = np.unique(pr.family, return_inverse=True)
    members = [np.where(idx == k)[0] for k in range(len(fams))]
    rng = np.random.default_rng(42); ba, bf = [], []
    for _ in range(5000):
        ii = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        ba.append((yt[ii] == yp[ii]).mean())
        bf.append(precision_recall_fscore_support(yt[ii], yp[ii], average="macro", zero_division=0)[2])
    train_f = set(man[man.split == "train"].family)
    fruit = lambda k: k.replace("fresh", "").replace("rotten", "")
    cond = lambda k: k[:5]
    w = pr[pr.label != pr.predicted]
    hist = pd.read_csv(arch / "training_history.csv")
    env = json.loads((arch / "S_Code_S2_environment.json").read_text())
    return dict(rr=rr, man=man, pr=pr, yt=yt, yp=yp, p=p, r=r, f=f, n=n, mp=mp, mr=mr, mf=mf,
                acc=(yt == yp).mean(), correct=int((yt == yp).sum()), ntest=len(yt),
                ci_acc=np.percentile(ba, [2.5, 97.5]), ci_f1=np.percentile(bf, [2.5, 97.5]),
                acc_orig=(yt[o] == yp[o]).mean(), n_orig=int(o.sum()), corr_orig=int((yt[o] == yp[o]).sum()), mf_orig=mfo,
                f_orig=fo, n_orig_cls=no,
                by_variant=pr.assign(c=pr.label == pr.predicted).groupby("variant").c.agg(["sum", "count", "mean"]),
                sib=int(pr.family.isin(train_f).sum()),
                cm=confusion_matrix(yt, yp, labels=range(6)),
                err=len(w), err_cond=int((w.label.map(fruit) == w.predicted.map(fruit)).sum()),
                err_fruit=int((w.label.map(fruit) != w.predicted.map(fruit)).sum()),
                acc_cond=(pr.label.str[:5] == pr.predicted.str[:5]).mean(),
                acc_fruit=(pr.label.map(fruit) == pr.predicted.map(fruit)).mean(),
                hist=hist, env=env,
                split_counts=man.groupby(["label", "split"]).size().unstack().reindex(CLASSES),
                fam_split=man.groupby(["label", "split"]).family.nunique().unstack().reindex(CLASSES))

I, F = analyse(IMG), analyse(FAM)
man = I["man"]
pct = lambda x, d=2: f"{100 * x:.{d}f}"
cnt = lambda x: f"{int(x):,}"

# ---- perceptual-hash recall against true photograph families
pairs = pd.read_csv(IMG / "S2_near_duplicate_pairs.csv")
fmap = man.set_index("relative_path").family
pairs["same"] = fmap.loc[pairs.file_a].values == fmap.loc[pairs.file_b].values
n_photos = man.family.nunique()
true_pairs = int(sum(k * (k - 1) // 2 for k in man.groupby("family").size()))
ph = {d: pairs[pairs.hamming <= d] for d in (0, 2, 4, 6)}

V = {}
V.update(RAW=cnt(I["rr"]["duplicate_audit"]["raw_files"]), UNIQUE=cnt(len(man)), PHOTOS=cnt(n_photos),
         EXACT=cnt(I["rr"]["duplicate_audit"]["exact_duplicate_files_removed"]),
         TRUEPAIRS=cnt(true_pairs), PH0=cnt(len(ph[0])), PH0SAME=pct(ph[0].same.mean(), 1),
         PH6=cnt(len(ph[6])), PH6SAME=pct(ph[6].same.mean(), 1),
         PHREC0=pct(ph[0].same.sum() / true_pairs, 1), PHREC6=pct(ph[6].same.sum() / true_pairs, 1),
         PH6CROSS=cnt(int(pairs[pairs.hamming <= 6].cross_label.sum())),
         PH0XFAM=cnt(int((~ph[0].same).sum())),
         RGBA=cnt((man["mode"] == "RGBA").sum()),
         WMIN=man.width.min(), WMAX=man.width.max(), HMIN=man.height.min(), HMAX=man.height.max(),
         KTRAIN=cnt((man.source_split == "train").sum()), KTEST=cnt((man.source_split == "test").sum()),
         DOWNLOADED=I["rr"]["provenance"]["downloaded_utc"][:10])
for tag, R in (("I", I), ("F", F)):
    sc = R["rr"]["split_counts"]
    V.update({f"{tag}_TRAIN": cnt(sc["train"]), f"{tag}_VAL": cnt(sc["validation"]), f"{tag}_TEST": cnt(sc["test"]),
              f"{tag}_ACC": pct(R["acc"]), f"{tag}_CORR": cnt(R["correct"]), f"{tag}_N": cnt(R["ntest"]),
              f"{tag}_CILO": pct(R["ci_acc"][0], 1), f"{tag}_CIHI": pct(R["ci_acc"][1], 1),
              f"{tag}_F1CILO": pct(R["ci_f1"][0], 1), f"{tag}_F1CIHI": pct(R["ci_f1"][1], 1),
              f"{tag}_MP": pct(R["mp"]), f"{tag}_MR": pct(R["mr"]), f"{tag}_MF": pct(R["mf"]),
              f"{tag}_ACCO": pct(R["acc_orig"]), f"{tag}_NO": cnt(R["n_orig"]), f"{tag}_CORRO": cnt(R["corr_orig"]),
              f"{tag}_MFO": pct(R["mf_orig"]),
              f"{tag}_SIB": cnt(R["sib"]), f"{tag}_SIBPCT": pct(R["sib"] / R["ntest"], 1),
              f"{tag}_ERR": cnt(R["err"]), f"{tag}_ERRCOND": cnt(R["err_cond"]), f"{tag}_ERRFRUIT": cnt(R["err_fruit"]),
              f"{tag}_ACCCOND": pct(R["acc_cond"]), f"{tag}_ACCFRUIT": pct(R["acc_fruit"]),
              f"{tag}_EPOCHS": R["rr"]["epochs_run"], f"{tag}_BEST": R["rr"]["best_epoch"],
              f"{tag}_MIN": f'{R["rr"]["training_minutes"]:.1f}',
              f"{tag}_BESTVAL": pct(R["hist"].val_accuracy.iloc[R["rr"]["best_epoch"] - 1]),
              f"{tag}_PHOTOS_TEST": cnt(R["man"][R["man"].split == "test"].family.nunique()),
              f"{tag}_PHOTOS_TRAIN": cnt(R["man"][R["man"].split == "train"].family.nunique()),
              f"{tag}_PHOTOS_VAL": cnt(R["man"][R["man"].split == "validation"].family.nunique())})
    hv = R["by_variant"]
    V[f"{tag}_VARMIN"] = VNAMES[hv["mean"].idxmin()]; V[f"{tag}_VARMINACC"] = pct(hv["mean"].min())
    V[f"{tag}_VARMAX"] = VNAMES[hv["mean"].idxmax()]; V[f"{tag}_VARMAXACC"] = pct(hv["mean"].max())
V["DIFF"] = f'{100 * (I["acc"] - F["acc"]):.2f}'
V["DIFF_F1"] = f'{100 * (I["mf"] - F["mf"]):.2f}'
V["DIFF_ORIG"] = f'{100 * (I["acc_orig"] - F["acc_orig"]):.2f}'
# worst classes and largest confusions for the family split
order = np.argsort(F["f"]); V["F_WORST"] = NAMES[order[0]].lower(); V["F_WORSTF1"] = pct(F["f"][order[0]])
V["F_WORST2"] = NAMES[order[1]].lower(); V["F_WORST2F1"] = pct(F["f"][order[1]])
V["F_BESTCLS"] = NAMES[order[-1]].lower(); V["F_BESTCLSF1"] = pct(F["f"][order[-1]])  # was F_BEST, which overwrote the photo-split best epoch
cmo = F["cm"].copy(); np.fill_diagonal(cmo, 0)
top = np.dstack(np.unravel_index(np.argsort(cmo.ravel())[::-1], cmo.shape))[0][:3]
V["F_CONF"] = "; ".join(f"{NAMES[i].lower()} predicted as {NAMES[j].lower()} ({cmo[i, j]})" for i, j in top)
e = F["env"]
V.update(GPU=e["gpus"][0]["device_name"] if e["gpus"] else "CPU", TF=e["tensorflow"], KERAS=e["keras"],
         PY=e["python"].split()[0], SKL=e["scikit_learn"], NP=e["numpy"], CUDA=e["cuda_build"], CUDNN=e["cudnn_build"])

# ---- LaTeX tables
rows = []
for k, c in enumerate(CLASSES):
    m = man[man.label == c]
    fs, ff = F["split_counts"].loc[c], F["fam_split"].loc[c]
    rows.append(f"{NAMES[k]} & {cnt(m.family.nunique())} & {cnt(len(m))} & {cnt(I['split_counts'].loc[c,'train'])}/{cnt(I['split_counts'].loc[c,'validation'])}/{cnt(I['split_counts'].loc[c,'test'])} & "
                f"{cnt(fs['train'])}/{cnt(fs['validation'])}/{cnt(fs['test'])} & {cnt(ff['test'])} \\\\")
rows.append("\\midrule")
rows.append(f"Total & {cnt(n_photos)} & {cnt(len(man))} & {V['I_TRAIN']}/{V['I_VAL']}/{V['I_TEST']} & {V['F_TRAIN']}/{V['F_VAL']}/{V['F_TEST']} & {V['F_PHOTOS_TEST']} \\\\")
V["TABLE_DATA"] = "\n".join(rows)

rows = []
for k in range(6):
    rows.append(f"{NAMES[k]} & {I['n'][k]} & {pct(I['p'][k])} & {pct(I['r'][k])} & {pct(I['f'][k])} & {F['n'][k]} & {pct(F['p'][k])} & {pct(F['r'][k])} & {pct(F['f'][k])} \\\\")
rows.append("\\midrule")
rows.append(f"Macro average & {I['ntest']:,} & {V['I_MP']} & {V['I_MR']} & {V['I_MF']} & {F['ntest']:,} & {V['F_MP']} & {V['F_MR']} & {V['F_MF']} \\\\")
V["TABLE_METRICS"] = "\n".join(rows)

rows = []
for v in VARIANTS:
    a, b = I["by_variant"].loc[v], F["by_variant"].loc[v]
    rows.append(f"{VNAMES[v]} & {int(a['sum'])}/{int(a['count'])} & {pct(a['mean'])} & {int(b['sum'])}/{int(b['count'])} & {pct(b['mean'])} \\\\")
V["TABLE_VARIANT"] = "\n".join(rows)

rows = []
for d in (0, 2, 4, 6):
    q = ph[d]
    rows.append(f"$\\leq${d} & {cnt(len(q))} & {cnt(int(q.same.sum()))} & {pct(q.same.mean(), 1)} & {pct(q.same.sum() / true_pairs, 1)} & {cnt(int(q.cross_label.sum()))} \\\\")
V["TABLE_PHASH"] = "\n".join(rows)

# ---- figures
import subprocess
subprocess.run([sys.executable, str(Path(__file__).parent / "make_figures.py"), str(IMG), str(FAM), str(PAPER / "Figures")], check=True)

if (PAPER / "manuscript_template.tex").exists():  # the v1 manuscript template is not part of the repository
    tpl = (PAPER / "manuscript_template.tex").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"<<([A-Z0-9_]+)>>", tpl)) - set(V))
    if missing:
        raise SystemExit(f"Template keys without values: {missing}")
    out = re.sub(r"<<([A-Z0-9_]+)>>", lambda m: str(V[m.group(1)]), tpl)
    (PAPER / "manuscript.tex").write_text(out, encoding="utf-8")
(PAPER / "numbers_used.json").write_text(json.dumps({k: v for k, v in V.items() if not k.startswith("TABLE")}, indent=2, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in V.items() if not k.startswith("TABLE")}, indent=1, ensure_ascii=False, default=str))
