"""Build the Supplementary Materials PDF (Tables S1-S3, Figure S1) from the v2 results folder and the per-pair table."""
import sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).parent; V2 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
TD = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE
R = pd.read_csv(TD / "TableS1_E3_pairs.csv"); C = pd.read_csv(TD / "TableS2_split_composition.csv")
man = pd.read_csv(V2 / "audit" / "manifest_v2.csv"); X = pd.read_csv(V2 / "audit" / "cross_family_tau0_pairs.csv")
# ---- Figure S1
fig, axs = plt.subplots(2, 2, figsize=(9, 6))
for col, tag, nm in ((0, "E3frozen", "Frozen VGG16"), (1, "E3ft", "Fine-tuned VGG16")):
    for f in sorted((V2 / "runs").glob(f"{tag}_s*_f*_*.history.csv")):
        h = pd.read_csv(f); ep = np.arange(1, len(h) + 1); arm = "exposed" if "exposed" in f.name else "clean"
        kw = dict(color="#2a78d6" if arm == "clean" else "#eb6834", lw=0.8, alpha=0.7)
        axs[0, col].plot(ep, h.val_loss, **kw); axs[1, col].plot(ep, 100 * h.val_accuracy, **kw)
    axs[0, col].set_title(nm); axs[0, col].set_ylabel("Validation loss"); axs[1, col].set_ylabel("Validation accuracy (%)")
    axs[1, col].set_xlabel("Epoch")
    for a in axs[:, col]: a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], color="#2a78d6"), Line2D([0], [0], color="#eb6834")], ["Clean arm", "Exposed arm"], ncol=2, loc="lower center", frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "FigureS1.png", dpi=200); plt.close(fig)
# ---- tables
f2 = lambda v: f"{v:.2f}"
def pfmt(p): return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"
rows = []
for _, r in R.iterrows():
    rows.append(f"{'Frozen' if r.backbone == 'frozen' else 'Fine-tuned'} & {r.seed} & {r.split} & {r.n_train:,} & {r.updates_clean:,} & {r.n_val:,} & {f2(r.val_exposed - r.val_clean)} & "
                f"{r.orig_n} & {f2(r.orig_acc_clean)} & {f2(r.orig_acc_exposed)} & {r.orig_only_exposed}/{r.orig_only_clean} & {pfmt(r.orig_mcnemar_p)} & {r.orig_boot_lo:+.2f} to {r.orig_boot_hi:+.2f} & "
                f"{f2(r.held_acc_clean)} & {f2(r.held_acc_exposed)} & {r.held_only_exposed}/{r.held_only_clean} \\\\")
S1 = "\n".join(rows)
rows = []
for _, r in C.iterrows():
    rows.append(f"{r.seed} & {r.split} & {r.base_files:,} & {r.base_families} & {r.donor_bundle_files:,} & {r.donor_bundle_families} & {r.exposure_bundle_files:,} & {r.exposure_bundle_families} & "
                f"{r.validation_files:,} & {r.validation_groups} & {r.eval_original_files} & {r.eval_original_groups} \\\\")
S2 = "\n".join(rows)
gid = {g: f"G{k + 1:04d}" for k, g in enumerate(sorted(man.group.unique()))}
fam = man.family.to_numpy(); grp = man.group.map(gid).to_numpy()
rows = []
tex = lambda t: t.replace("_", "\\_").replace("Screen Shot ", "")
for _, r in X.iterrows():
    rows.append(f"{tex(r.file_i.split('/')[-1])} & {tex(r.file_j.split('/')[-1])} & {man.label[r.i].replace('fresh','fresh ').replace('rotten','rotten ')} & {grp[r.i]} \\\\")
S3 = "\n".join(rows)
doc = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.6cm,landscape]{geometry}
\usepackage{booktabs,longtable,graphicx,amsmath}
\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\begin{center}\Large Supplementary Materials\\[2pt]\normalsize One Photograph, Nine Files: Provenance Audit and Sibling-Controlled Evaluation of a Public Fresh and Rotten Fruit Image Dataset\\ Ahmad Shalaldeh\end{center}
\small
\begin{longtable}{lrrrrrrrrrrrrrrr}
\caption{Every fit pair of the matched study (E3). Updates are equal in both arms. Val.~diff.: exposed minus clean validation accuracy (pp). Disc.: files correct only in the exposed arm / only in the clean arm. $p$: exact two-sided McNemar. Bootstrap: 95\% percentile interval of the original-capture difference (pp), test capture groups resampled, 5000 draws.}\\
\toprule
Backbone & Seed & Split & Train & Updates & Val. & Val. diff. & $n$ & Orig. clean & Orig. exp. & Disc. & $p$ & Bootstrap & Held clean & Held exp. & Disc. \\
\midrule\endhead
""" + S1 + r"""
\bottomrule
\end{longtable}

\begin{longtable}{rrrrrrrrrrrr}
\caption{Composition of every split (files and families by role). The base is common to both arms; donor bundles enter the clean arm only and exposure bundles the exposed arm only. Validation and evaluation counts are files and capture groups.}\\
\toprule
Seed & Split & Base files & Base families & Donor files & Donor families & Exposure files & Exposure families & Val. files & Val. groups & Eval. originals & Eval. groups \\
\midrule\endhead
""" + S2 + r"""
\bottomrule
\end{longtable}

\begin{longtable}{p{8.5cm}p{8.5cm}p{2.6cm}r}
\caption{Cross-family zero-distance perceptual-hash links (file pairs). All pairs were within one class and had different image dimensions. The last column is the identifier of the capture group that the merge produced (numbered in the order of the released manifest's sorted group keys).}\\
\toprule
File A & File B & Class & Group \\
\midrule\endhead
""" + S3 + r"""
\bottomrule
\end{longtable}

\begin{figure}[h]\centering\includegraphics[width=0.85\linewidth]{FigureS1.png}
\caption{Validation loss and accuracy of every E3 fit (blue, clean arm; orange, exposed arm). Final weights were evaluated at the end of the fixed budget.}\end{figure}
\end{document}
"""
(OUT / "supplementary.tex").write_text(doc)
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplementary.tex"], cwd=OUT, stdout=subprocess.DEVNULL)
print("supplement done", (OUT / "supplementary.pdf").exists())
