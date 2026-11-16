"""Additional v2 analyses requested in review: class-wise and version-wise errors, macro-F1, pooled discordance,
validation differences, training-curve checks, cross-family merge topology and per-pair supplementary tables.
All values are read from the v2 results folder; nothing is retrained."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from design_v2 import CLASSES, NAMES, DERIVED, paired_cluster_bootstrap

VNAME = {"rotated_by_15": "Rotation 15\\textdegree", "rotated_by_30": "Rotation 30\\textdegree", "rotated_by_45": "Rotation 45\\textdegree",
         "rotated_by_60": "Rotation 60\\textdegree", "rotated_by_75": "Rotation 75\\textdegree", "saltandpepper": "Salt-and-pepper copy",
         "translation": "Translated copy", "vertical_flip": "Vertical flip"}


def fmt_p(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e"); return f"{m} \\times 10^{{{int(e)}}}"
    return f"{p:.3f}" if p >= 0.0095 else f"{p:.4f}"


def pairs_of(V2, tag):
    out = []
    for f in sorted((V2 / "runs").glob(f"{tag}_s*_f*_clean.json")):
        pid = f.name[: -len("_clean.json")]
        s = int(pid.split("_s")[1].split("_")[0]); k = int(pid.split("_f")[1])
        out.append((pid, s, k))
    return out


def load(V2, rid, ep):
    return pd.read_csv(V2 / "runs" / f"{rid}.{ep}.csv").sort_values("row").reset_index(drop=True)


def compute(V2: Path, outdir: Path) -> dict:
    V = {}
    man = pd.read_csv(V2 / "audit" / "manifest_v2.csv")
    # ---------------- cross-family merge topology
    x = pd.read_csv(V2 / "audit" / "cross_family_tau0_pairs.csv")
    fam = man.family.to_numpy()
    edges = {tuple(sorted((fam[i], fam[j]))) for i, j in zip(x.i, x.j)}
    gsz = man.groupby("group").family.nunique().value_counts().sort_index()
    V["XFAM_EDGES"] = len(edges)
    V["GSIZE"] = (f"{gsz.get(1, 0):,} groups of one family, {gsz.get(2, 0)} of two" +
                  (f" and {gsz.get(3, 0)} of three" if gsz.get(3, 0) else ""))
    V["XSAMECLASS"] = bool((man.label.to_numpy()[x.i] == man.label.to_numpy()[x.j]).all())
    # ---------------- E3 per pair
    reg = []
    for tag in ("E3frozen", "E3ft"):
        for pid, s, k in pairs_of(V2, tag):
            rc = json.loads((V2 / "runs" / f"{pid}_clean.json").read_text()); rx = json.loads((V2 / "runs" / f"{pid}_exposed.json").read_text())
            row = {"backbone": "frozen" if tag == "E3frozen" else "fine-tuned", "seed": s, "split": k, "pair": pid,
                   "n_train": rc["n_train"], "updates_clean": rc["updates"], "updates_exposed": rx["updates"],
                   "n_val": rc["eval"]["val"]["n"],
                   "val_clean": 100 * rc["eval"]["val"]["correct"] / rc["eval"]["val"]["n"],
                   "val_exposed": 100 * rx["eval"]["val"]["correct"] / rx["eval"]["val"]["n"]}
            for ep in ("orig", "held"):
                a = load(V2, pid + "_clean", ep); b = load(V2, pid + "_exposed", ep)
                assert (a.row.values == b.row.values).all()
                ca = (a.label == a.pred).to_numpy(); cb = (b.label == b.pred).to_numpy()
                n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
                p = 1.0 if n01 + n10 == 0 else binomtest(n10, n01 + n10, 0.5).pvalue
                ci = paired_cluster_bootstrap(pd.DataFrame({"cluster": a.group.values, "a": cb.astype(int), "b": ca.astype(int)}))
                row.update({f"{ep}_n": len(a), f"{ep}_acc_clean": 100 * ca.mean(), f"{ep}_acc_exposed": 100 * cb.mean(),
                            f"{ep}_diff": 100 * (cb.mean() - ca.mean()), f"{ep}_only_clean": n01, f"{ep}_only_exposed": n10,
                            f"{ep}_mcnemar_p": p, f"{ep}_boot_lo": ci[0], f"{ep}_boot_hi": ci[1],
                            f"{ep}_groups": a.group.nunique(),
                            f"{ep}_macroF1_clean": 100 * f1_score(a.label, a.pred, average="macro", labels=range(6)),
                            f"{ep}_macroF1_exposed": 100 * f1_score(b.label, b.pred, average="macro", labels=range(6))})
            reg.append(row)
    R = pd.DataFrame(reg); R.to_csv(outdir / "TableS1_E3_pairs.csv", index=False)
    V["_REG"] = R
    for tag, key in (("frozen", "E3"), ("fine-tuned", "E3FT")):
        d = R[R.backbone == tag]
        if d.empty: continue
        V[f"{key}_NSEEDS"] = d.seed.nunique()
        V[f"{key}_DISC_MAX"] = int((d.orig_only_clean + d.orig_only_exposed).max())
        seeds = []
        for s, g in d.groupby("seed"):
            a, b = int(g.orig_only_clean.sum()), int(g.orig_only_exposed.sum())
            p = binomtest(b, a + b, 0.5).pvalue if a + b else 1.0
            seeds.append((s, b, a, p))
        V[f"{key}_SEEDS"] = seeds
        V[f"{key}_CORR"] = int(d.orig_only_exposed.sum()); V[f"{key}_INTRO"] = int(d.orig_only_clean.sum())
        vd = d.val_exposed - d.val_clean
        V[f"{key}_VALMEAN"] = f"{vd.mean():+.2f}".replace("-", "$-$"); V[f"{key}_VALSD"] = f"{vd.std(ddof=1):.2f}" if len(vd) > 1 else "n/a"
        V[f"{key}_VALPOS"] = int((vd > 0).sum()); V[f"{key}_VALNEG"] = int((vd < 0).sum())
        V[f"{key}_NPOS"] = int((d.orig_diff > 0).sum())
        V[f"{key}_VALN"] = f"{int(d.n_val.min()):,}--{int(d.n_val.max()):,}"
        V[f"{key}_UPDEQ"] = bool((d.updates_clean == d.updates_exposed).all())
        V[f"{key}_F1C"] = f"{d.orig_macroF1_clean.mean():.2f}"; V[f"{key}_F1E"] = f"{d.orig_macroF1_exposed.mean():.2f}"
        V[f"{key}_NFIT"] = len(d)
    # ---------------- class-wise and version-wise errors
    def pooled(tag, ep, arm):
        return pd.concat([load(V2, f"{pid}_{arm}", ep) for pid, _, _ in pairs_of(V2, tag)])
    cls_rows = []
    have_ft = bool(pairs_of(V2, "E3ft"))
    tabs = {}
    for tag in ("E3frozen", "E3ft"):
        if not pairs_of(V2, tag): continue
        for arm in ("clean", "exposed"):
            tabs[(tag, arm, "orig")] = pooled(tag, "orig", arm); tabs[(tag, arm, "held")] = pooled(tag, "held", arm)
    for k, nm in enumerate(NAMES):
        cells = []
        for tag in ("E3frozen", "E3ft"):
            if (tag, "clean", "orig") not in tabs: continue
            c = tabs[(tag, "clean", "orig")]; e = tabs[(tag, "exposed", "orig")]
            n = int((c.label == k).sum())
            cells += [f"{n:,}", str(int(((c.label == k) & (c.pred != k)).sum())), str(int(((e.label == k) & (e.pred != k)).sum()))]
        cls_rows.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    tot = []
    for tag in ("E3frozen", "E3ft"):
        if (tag, "clean", "orig") not in tabs: continue
        c = tabs[(tag, "clean", "orig")]; e = tabs[(tag, "exposed", "orig")]
        tot += [f"{len(c):,}", str(int((c.label != c.pred).sum())), str(int((e.label != e.pred).sum()))]
    cls_rows += ["\\midrule", "Total & " + " & ".join(tot) + " \\\\"]
    V["TABLE_CLASS"] = "\n".join(cls_rows)
    ver_rows = []
    for v in DERIVED:
        cells = []
        for tag in ("E3frozen", "E3ft"):
            if (tag, "clean", "held") not in tabs: continue
            c = tabs[(tag, "clean", "held")]; e = tabs[(tag, "exposed", "held")]
            cm, em = c[c.variant == v], e[e.variant == v]
            cells += [f"{len(cm):,}", str(int((cm.label != cm.pred).sum())), str(int((em.label != em.pred).sum()))]
        ver_rows.append(f"{VNAME[v]} & " + " & ".join(cells) + " \\\\")
    V["TABLE_VERSION"] = "\n".join(ver_rows)
    # class-error narrative values (frozen, originals)
    c = tabs[("E3frozen", "clean", "orig")]; e = tabs[("E3frozen", "exposed", "orig")]
    errc = {NAMES[k]: int(((c.label == k) & (c.pred != k)).sum()) for k in range(6)}
    erre = {NAMES[k]: int(((e.label == k) & (e.pred != k)).sum()) for k in range(6)}
    V["CLS_TOP"] = sorted(errc, key=errc.get, reverse=True)[:2]
    V["CLS_ZERO"] = [k for k in errc if errc[k] == 0]
    V["CLS_ALLDOWN"] = all(erre[k] <= errc[k] for k in errc) and all(erre[k] < errc[k] for k in errc if errc[k] > 0)
    # version narrative
    vt = {}
    for tag in ("E3frozen", "E3ft"):
        if (tag, "clean", "held") not in tabs: continue
        c = tabs[(tag, "clean", "held")]; e = tabs[(tag, "exposed", "held")]
        vt[tag] = {v: (int(((c.variant == v) & (c.label != c.pred)).sum()), int(((e.variant == v) & (e.label != e.pred)).sum())) for v in DERIVED}
    V["_VT"] = vt
    # ---------------- training curves
    def curves(tag):
        H = [pd.read_csv(f) for f in sorted((V2 / "runs").glob(f"{tag}_s*_f*_*.history.csv"))]
        return np.array([h.val_loss.values for h in H]), np.array([h.val_accuracy.values for h in H])
    L, A = curves("E3frozen")
    am = np.argmin(L, 1) + 1
    V["FZ_MINEP"] = f"{am.min()}--{am.max()}"; V["FZ_L20"] = f"{L[:, 19].mean():.3f}"; V["FZ_L30"] = f"{L[:, 29].mean():.3f}"
    V["FZ_A20"] = f"{100 * A[:, 19].mean():.2f}"; V["FZ_A30"] = f"{100 * A[:, 29].mean():.2f}"
    if have_ft:
        L, A = curves("E3ft")
        am = np.argmin(L, 1) + 1
        V["FT_MINEP"] = f"{am.min()}--{am.max()}"; V["FT_L9"] = f"{L[:, 8].mean():.3f}"; V["FT_L12"] = f"{L[:, 11].mean():.3f}"
        V["FT_A9"] = f"{100 * A[:, 8].mean():.2f}"; V["FT_A12"] = f"{100 * A[:, 11].mean():.2f}"
    # ---------------- E4/E5 macro-F1
    V["_F1"] = {}
    for f in sorted((V2 / "runs").glob("E[45]*.test.csv")):
        d = pd.read_csv(f); V["_F1"][f.name.split(".")[0]] = 100 * f1_score(d.label, d.pred, average="macro", labels=range(6))
    # ---------------- per-split composition (supplementary)
    comp = []
    for f in sorted((V2 / "designs").glob("design_s*_f*.csv")):
        d = pd.read_csv(f).merge(man[["relative_path", "label", "family", "group", "variant"]], on="relative_path")
        s, k = f.stem.split("_")[1][1:], f.stem.split("_")[2][1:]
        r = {"seed": int(s), "split": int(k)}
        for role in ("base", "donor_bundle", "exposure_bundle", "validation", "eval_original", "eval_withheld"):
            q = d[d.role == role]; r[f"{role}_files"] = len(q); r[f"{role}_families"] = q.family.nunique(); r[f"{role}_groups"] = q.group.nunique()
        comp.append(r)
    C = pd.DataFrame(comp); C.to_csv(outdir / "TableS2_split_composition.csv", index=False); V["_COMP"] = C
    return V
