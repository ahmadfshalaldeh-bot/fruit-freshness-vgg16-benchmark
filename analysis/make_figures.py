# HISTORICAL (v1.0). This script reproduces the earlier single-split analysis; its bootstrap resamples individual
# files, not capture groups. The v2 analysis (capture-group bootstrap, matched design) is in v2/analysis/.
"""Manuscript figures, drawn only from archive outputs (no dataset photographs are reproduced).
Usage: python make_figures.py <image_split_archive> <family_split_archive> <out_dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

IMG, FAM, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); O.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#f0efec"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "savefig.dpi": 600})
CLASSES = ["freshapples", "rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges"]
SHORT = ["Fresh\napple", "Rotten\napple", "Fresh\nbanana", "Rotten\nbanana", "Fresh\norange", "Rotten\norange"]
OP_RE = r"^(rotated_by_\d+|saltandpepper|translation|vertical_flip)_"
VARIANTS = ["original", "rotated_by_15", "rotated_by_30", "rotated_by_45", "rotated_by_60", "rotated_by_75",
            "saltandpepper", "translation", "vertical_flip"]
VSHORT = ["Original", "Rot. 15°", "Rot. 30°", "Rot. 45°", "Rot. 60°", "Rot. 75°", "Salt &\npepper", "Transl.", "Vertical\nflip"]
ri, rf = (json.loads((a / "run_record.json").read_text()) for a in (IMG, FAM))

def box(ax, x, y, w, h, title, body, fc=SURF):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=INK2, lw=0.9))
    nt = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top", fontsize=8, weight="bold", color=INK, linespacing=1.15)
    ax.text(x + w / 2, y + h - 0.05 - 0.09 * nt - 0.03, body, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.4)
def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=INK2))

# Figure 1: workflow
a = ri["duplicate_audit"]; si, sf = ri["split_counts"], rf["split_counts"]
fig, ax = plt.subplots(figsize=(9.0, 2.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
w, h, y = 0.175, 0.62, 0.30; xs = [0.005, 0.205, 0.405, 0.605, 0.805]
steps = [("1. Acquire", f"Kaggle download\n{a['raw_files']:,} files\nsix source labels\ndate and path logged"),
         ("2. Provenance audit", f"SHA-256: {a['exact_duplicate_files_removed']:,} exact\ncopies removed\nfile names: {a['analysis_images']:,}\nimages = 1,511 photos × 9"),
         ("3. Split (seed 42)", f"Image-level:\n{si['train']:,} / {si['validation']:,} / {si['test']:,}\nPhoto-level:\n{sf['train']:,} / {sf['validation']:,} / {sf['test']:,}"),
         ("4. Train", "224×224 RGB, [0, 1]\ntrain-only augmentation\nfrozen VGG16 + head\nAdam, ≤100 epochs"),
         ("5. Evaluate", "held-out test only\nconfusion matrix, P/R/F1\nphoto-clustered\nbootstrap 95% CI")]
for i, (t, b) in enumerate(steps):
    box(ax, xs[i], y, w, h, t, b, fc="#e8f1fc" if i in (1, 2) else SURF)
    if i < 4: arrow(ax, xs[i] + w + 0.003, y + h / 2, xs[i + 1] - 0.003, y + h / 2)
ax.text(0.5, 0.17, "Each step writes to the reproducibility archive: file manifest with checksums and photo identifiers, duplicate audit,\n"
        "held-out predictions with class probabilities, environment lock file, model summary and training log.",
        ha="center", va="top", fontsize=7.2, color=INK2, style="italic")
fig.savefig(O / "Figure1_workflow.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# Figure 2: architecture
layers = [("Input\nimage", "224×224×3 RGB\nscaled to [0, 1]", SURF, "0"),
          ("VGG16\nconv. base", "13 conv. layers\n5 blocks, ImageNet\nweights, frozen\nout: 7×7×512", "#dcdcd8", "14,714,688\n(frozen)"),
          ("Global average\npooling", "out: 512", "#e8f1fc", "0"),
          ("Dense 256\nReLU", "+ Dropout 0.5", "#e8f1fc", "131,328"),
          ("Dense 128\nReLU", "+ Dropout 0.5", "#e8f1fc", "32,896"),
          ("Dense 6\nsoftmax", "six class\nprobabilities", "#e8f1fc", "774")]
fig, ax = plt.subplots(figsize=(9.0, 2.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
w, h, y = 0.15, 0.58, 0.30; gap = (1 - 6 * w) / 5
for i, (t, b, fc, p) in enumerate(layers):
    x = i * (w + gap); box(ax, x, y, w, h, t, b, fc=fc)
    ax.text(x + w / 2, y - 0.05, f"params: {p}", ha="center", va="top", fontsize=6.6, color=INK2)
    if i < 5: arrow(ax, x + w + 0.002, y + h / 2, x + w + gap - 0.002, y + h / 2)
ax.text(0.5, 0.03, "Total parameters 14,879,686 · trainable 164,998 (classification head only) · categorical cross-entropy, Adam (lr = 1×10⁻⁴), batch 32",
        ha="center", fontsize=7.2, color=INK)
fig.savefig(O / "Figure2_architecture.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# Figure 3: schematic of how the two split protocols treat the 9 stored versions of a photograph
PC = {"train": "#9ec5f4", "validation": "#f3c49e", "test": "#eb6834"}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5))
rng = np.random.default_rng(7)
for ax, title, mode in ((axs[0], "(a) Image-level split", "image"), (axs[1], "(b) Photo-level split", "photo")):
    ax.set_xlim(0, 9.4); ax.set_ylim(-0.6, 4.2); ax.axis("off"); ax.set_title(title)
    for r in range(4):
        if mode == "image":
            labs = rng.choice(["train", "validation", "test"], 9, p=[0.7, 0.15, 0.15])
        else:
            labs = [["train", "train", "validation", "test"][r]] * 9
        for c in range(9):
            ax.add_patch(Rectangle((c + 0.3, 3.2 - r), 0.85, 0.75, fc=PC[labs[c]], ec="white", lw=1.5))
        ax.text(0.15, 3.57 - r, f"P{r + 1}", ha="right", va="center", fontsize=7.5, color=INK2)
    for c, lab in enumerate(["O", "15", "30", "45", "60", "75", "SP", "T", "VF"]):
        ax.text(c + 0.725, -0.25, lab, ha="center", fontsize=6.5, color=INK2)
handles = [Rectangle((0, 0), 1, 1, fc=PC[k]) for k in PC]
fig.legend(handles, ["Training", "Validation", "Test"], ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=7.6)
fig.text(0.5, -0.12, "Rows P1–P4: four example photographs. Columns: stored versions (O original; 15–75 rotation in degrees; SP salt-and-pepper; T translation; VF vertical flip).",
         ha="center", fontsize=6.6, color=INK2)
fig.tight_layout(); fig.savefig(O / "Figure3_split_schematic.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# Figure 4: photo-level confusion matrix + class F1 under both protocols
cm = pd.read_csv(FAM / "confusion_matrix_counts.csv", index_col=0).to_numpy()
ti = pd.read_csv(IMG / "Table_3_class_metrics.csv").iloc[:6]; tf = pd.read_csv(FAM / "Table_3_class_metrics.csv").iloc[:6]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1, 1.1]})
rown = cm / cm.sum(1, keepdims=True)
cmap = ListedColormap(["#fcfcfb"] + BLUES); norm = BoundaryNorm([0, 0.005, 0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001], cmap.N)
ax[0].imshow(rown, cmap=cmap, norm=norm)
for i in range(6):
    for j in range(6):
        v = cm[i, j]
        ax[0].text(j, i, v, ha="center", va="center", fontsize=8, color="white" if rown[i, j] > 0.5 else (INK if v else "#b0afaa"))
ax[0].set_xticks(range(6), SHORT, fontsize=6.8); ax[0].set_yticks(range(6), SHORT, fontsize=6.8)
ax[0].set_xlabel("Predicted label"); ax[0].set_ylabel("True label"); ax[0].tick_params(length=0)
for s in ax[0].spines.values(): s.set_visible(False)
ax[0].set_title(f"(a) Photo-level split, n = {int(cm.sum()):,}")
x = np.arange(6); bw = 0.36
ax[1].bar(x - bw / 2, ti["F1-score (%)"], bw * 0.92, color=S1, label="Image-level split", zorder=3)
ax[1].bar(x + bw / 2, tf["F1-score (%)"], bw * 0.92, color=S2, label="Photo-level split", zorder=3)
lo = np.floor(min(ti["F1-score (%)"].min(), tf["F1-score (%)"].min()) / 5) * 5 - 5
ax[1].set_ylim(max(lo, 0), 100.8); ax[1].set_xticks(x, SHORT, fontsize=6.8); ax[1].set_ylabel("F1-score (%)")
ax[1].yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax[1].spines[["top", "right"]].set_visible(False)
ax[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.6)
ax[1].set_title("(b) Class F1-score by split protocol")
fig.tight_layout(); fig.savefig(O / "Figure4_results.png", facecolor="white", bbox_inches="tight"); plt.close(fig)

# Figure 5: accuracy by stored version
def byvar(arch):
    p = pd.read_csv(arch / "S3_heldout_predictions.csv"); fn = p.relative_path.str.split("/").str[-1]
    p["variant"] = fn.str.extract(OP_RE)[0].fillna("original")
    return p.assign(c=p.label == p.predicted).groupby("variant").c.mean().reindex(VARIANTS) * 100
vi, vf = byvar(IMG), byvar(FAM)
fig, ax = plt.subplots(figsize=(7.2, 2.8)); x = np.arange(9)
ax.bar(x - bw / 2, vi, bw * 0.92, color=S1, label="Image-level split", zorder=3)
ax.bar(x + bw / 2, vf, bw * 0.92, color=S2, label="Photo-level split", zorder=3)
lo = np.floor(min(vi.min(), vf.min()) / 5) * 5 - 5
ax.set_ylim(max(lo, 0), 100.8); ax.set_xticks(x, VSHORT, fontsize=7); ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.6)
fig.tight_layout(); fig.savefig(O / "Figure5_by_version.png", facecolor="white", bbox_inches="tight"); plt.close(fig)

# Figure 6: training curves (photo-level run, with image-level validation for reference)
hf, hi = pd.read_csv(FAM / "training_history.csv"), pd.read_csv(IMG / "training_history.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
for a_, (tr, va, yl, tt) in zip(ax, [("loss", "val_loss", "Categorical cross-entropy", "(a) Loss"),
                                     ("accuracy", "val_accuracy", "Accuracy", "(b) Accuracy")]):
    a_.plot(hf.epoch + 1, hf[tr], color=S2, lw=1.3, label="Photo-level: training")
    a_.plot(hf.epoch + 1, hf[va], color=S2, lw=1.3, ls="--", label="Photo-level: validation")
    a_.plot(hi.epoch + 1, hi[va], color=S1, lw=1.3, ls="--", label="Image-level: validation")
    a_.axvline(rf["best_epoch"], color=INK2, ls=":", lw=0.8)
    a_.set_xlabel("Epoch"); a_.set_ylabel(yl); a_.set_title(tt)
    a_.yaxis.grid(True, color=GRID, lw=0.6); a_.spines[["top", "right"]].set_visible(False)
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=7.4)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(O / "Figure6_training.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
print("figures done")
