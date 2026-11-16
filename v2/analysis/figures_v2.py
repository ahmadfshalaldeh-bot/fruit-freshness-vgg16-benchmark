"""Figures for the v2 manuscript (no dataset photographs are reproduced).
Usage: python figures_v2.py <v2_results_dir or 'none'> <out_dir>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

V2 = None if sys.argv[1] == "none" else Path(sys.argv[1]); O = Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).parent
SUMD = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#f0efec"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "savefig.dpi": 600})

def box(ax, x, y, w, h, title, body, fc=SURF, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.01", fc=fc, ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.8, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.115, body, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.35)
def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK2))

# ---------------- Figure 1: audit and matched design
fig = plt.figure(figsize=(9.2, 4.6))
ax = fig.add_axes([0, 0.58, 1, 0.40]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.98, "(a) Provenance audit", fontsize=9.5, weight="bold", va="top")
steps = [("SHA-256 checksums", "27,198 files\n→ 13,599 unique\n(mirrored folder)"),
         ("File-name parsing", "13,599 files\n→ 1,511 families\n× 9 stored versions"),
         ("Perceptual hash (τ = 0)", "29 cross-family links\n→ 1,498 capture\ngroups"),
         ("Kaggle folders", "1,310 families span\ntrain and test folders"),
         ("Detectors vs. truth", "pHash and VGG16\nembeddings scored\nagainst families")]
w, h, y = 0.178, 0.58, 0.12
for i, (t, b) in enumerate(steps):
    x = 0.005 + i * 0.2
    box(ax, x, y, w, h, t, b, fc="#e8f1fc" if i in (1, 2) else SURF)
    if i < 4: arrow(ax, x + w + 0.003, y + h / 2, x + 0.2 - 0.003, y + h / 2)
ax2 = fig.add_axes([0.02, 0.02, 0.96, 0.52]); ax2.set_xlim(0, 10); ax2.set_ylim(0, 5); ax2.axis("off")
ax2.text(0, 4.95, "(b) Matched contamination design (one held-out capture and its donor)", fontsize=9.5, weight="bold", va="top")
VER = ["O", "R15", "R30", "R45", "R60", "R75", "SP", "T", "VF"]
C_TEST, C_HELD, C_TRAIN, C_OFF = "#eb6834", "#f3c49e", "#2a78d6", "#e6e5e1"
def row(y, label, colors, note):
    ax2.text(0.05, y + 0.3, label, fontsize=7.6, va="center", ha="left", color=INK)
    for k, (v, col) in enumerate(zip(VER, colors)):
        ax2.add_patch(Rectangle((2.3 + k * 0.52, y), 0.46, 0.6, fc=col, ec="white", lw=1.2))
        ax2.text(2.53 + k * 0.52, y + 0.3, v, ha="center", va="center", fontsize=5.6, color="white" if col in (C_TEST, C_TRAIN) else INK)
    ax2.text(7.15, y + 0.3, note, fontsize=7.2, va="center", color=INK2)
held = 3  # example withheld version (45°)
test_clean = [C_TEST] + [C_HELD if k == held else C_OFF for k in range(1, 9)]
donor_clean = [C_OFF] + [C_OFF if k == held else C_TRAIN for k in range(1, 9)]
test_exp = [C_TEST] + [C_HELD if k == held else C_TRAIN for k in range(1, 9)]
donor_exp = [C_OFF] * 9
ax2.text(0.05, 4.55, "Both arms also train on the same common base (all nine files of every eligible non-donor family); validation groups are never trained on.", fontsize=7.0, color=INK2, va="center")
ax2.text(0.05, 4.15, "Clean arm", fontsize=8.2, weight="bold")
row(3.35, "Held-out capture", test_clean, "evaluated: O and withheld version")
row(2.6, "Donor (same class)", donor_clean, "7 versions in training")
ax2.text(0.05, 1.95, "Exposed arm", fontsize=8.2, weight="bold")
row(1.15, "Held-out capture", test_exp, "7 versions in training; same files evaluated")
row(0.4, "Donor (same class)", donor_exp, "not used")
handles = [Rectangle((0, 0), 1, 1, fc=cc) for cc in (C_TEST, C_HELD, C_TRAIN, C_OFF)]
ax2.legend(handles, ["Evaluated original", "Evaluated withheld version", "In training", "Not used"], ncol=4, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.17), fontsize=7.2)
fig.savefig(O / "Figure1_design.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

if V2 is not None:
    # ---------------- Figure 2: embedding similarity
    sib = V2 / "embed" / "sim_sibling.npy"
    if sib.exists():
        s = np.load(sib).astype(float); n = np.load(V2 / "embed" / "sim_nonsibling_sample.npy").astype(float)
        fig, ax = plt.subplots(figsize=(6.6, 2.8)); bins = np.linspace(min(s.min(), n.min()), 1, 80)
        ax.hist(n, bins=bins, density=True, color=S1, alpha=0.75, label="Non-sibling pairs (same class)")
        ax.hist(s, bins=bins, density=True, color=S2, alpha=0.75, label="Sibling pairs")
        ax.set_xlabel("Cosine similarity of VGG16 embeddings"); ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7.6, loc="upper left")
        fig.tight_layout(); fig.savefig(O / "Figure2_embedding.png", facecolor="white"); plt.close(fig)
    # ---------------- Figure 3: matched differences
    parts = []
    for tag, nm in (("E3frozen", "Frozen VGG16"), ("E3ft", "Fine-tuned VGG16")):
        p = SUMD / f"{tag}_pairs_summary.csv"
        if p.exists():
            d = pd.read_csv(p, converters={"orig_ci": eval, "held_ci": eval}); d["model"] = nm; parts.append(d)
    if parts:
        d = pd.concat(parts, ignore_index=True)
        d = d.assign(seed=d.pair.str.extract(r"_s(\d+)_")[0].astype(int), split=d.pair.str.extract(r"_f(\d+)$")[0].astype(int))
        fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
        for ax, ep, tt in ((axs[0], "orig", "(a) Original captures"), (axs[1], "held", "(b) Withheld derived version")):
            ticks, labels, x0 = [], [], 0
            for k, (nm, g) in enumerate(d.groupby("model", sort=False)):
                g = g.sort_values(["seed", "split"])
                xs = np.arange(len(g)) + x0; x0 += len(g) + 1
                lo = g[f"{ep}_diff"] - g[f"{ep}_ci"].str[0]; hi = g[f"{ep}_ci"].str[1] - g[f"{ep}_diff"]
                ax.errorbar(xs, g[f"{ep}_diff"], yerr=[lo, hi], fmt="o" if k == 0 else "s", ms=4.5, color=S1 if k == 0 else S2,
                            ecolor=INK2, elinewidth=1.0, capsize=2, label=nm)
                ticks += list(xs); labels += [f"S{a}F{b}" for a, b in zip(g.seed, g.split)]
            ax.axhline(0, color=INK2, lw=0.8); ax.set_title(tt)
            ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=90, fontsize=6.2)
            ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color=GRID, lw=0.5)
        axs[0].set_ylabel("Exposed − clean accuracy (pp)")
        h_, l_ = axs[0].get_legend_handles_labels()
        fig.legend(h_, l_, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7.6)
        fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(O / "Figure3_matched.png", dpi=300, facecolor="white"); plt.close(fig)
print("figures done")
