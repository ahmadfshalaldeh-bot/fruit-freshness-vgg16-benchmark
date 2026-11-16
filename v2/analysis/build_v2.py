"""Fill the manuscript from archived results.

Usage: python build_v2.py <v1_image_archive> <v1_photo_archive> [<v2_results_dir>]
Values that need the v2 run are shown as highlighted [pending] boxes until the v2 folder is given.
"""
from sklearn.metrics import f1_score
import sys, json, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent / "v2"), str(HERE.parent / "src")]
from design_v2 import (CLASSES, NAMES, VARIANTS, add_provenance, UnionFind, mcnemar_exact,
                       paired_cluster_bootstrap, t_interval)

V1I, V1P = Path(sys.argv[1]), Path(sys.argv[2])
V2 = Path(sys.argv[3]) if len(sys.argv) > 3 and Path(sys.argv[3]).exists() else None
OUTD = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE; OUTD.mkdir(parents=True, exist_ok=True)
V = {}
pend = lambda k: f"\\pend{{{k}}}"
c = lambda x: f"{int(x):,}"
pc = lambda x, d=2: f"{100 * x:.{d}f}"

# ------------------------------------------------------------ E1 provenance (v1 archives suffice)
rr = json.loads((V1I / "run_record.json").read_text())
man = add_provenance(pd.read_csv(V1P / "S1_manifest_with_split.csv"))
pairs = pd.read_csv(V1I / "S2_near_duplicate_pairs.csv")
fam = man.set_index("relative_path").family; var = man.set_index("relative_path").variant
pairs["same"] = fam.loc[pairs.file_a].values == fam.loc[pairs.file_b].values
pairs["vp"] = ["|".join(sorted(x)) for x in zip(var.loc[pairs.file_a].values, var.loc[pairs.file_b].values)]
true_pairs = int(sum(k * (k - 1) // 2 for k in man.groupby("family").size()))
trf = set(man[man.source_split == "train"].family); te = man[man.source_split == "test"]
famsrc = man.groupby("family").source_split.agg(lambda s: "+".join(sorted(set(s))))
q6 = pairs[(pairs.hamming <= 6) & pairs.same]
easy = q6.vp.isin(["original|saltandpepper", "original|vertical_flip", "saltandpepper|vertical_flip"]).sum()
rot = q6.vp.str.contains("rotated").sum(); tr_ = q6.vp.str.contains("translation").sum()
uf = UnionFind(man.relative_path.tolist())
for a, b in zip(q6.file_a, q6.file_b): uf.union(a, b)
comps = man.assign(cc=man.relative_path.map(uf.find)).groupby("family").cc.nunique()
x0 = pairs[(pairs.hamming == 0) & ~pairs.same]
uf2 = UnionFind(man.family.unique().tolist())
for a, b in zip(fam.loc[x0.file_a], fam.loc[x0.file_b]): uf2.union(a, b)
groups = man.family.map(uf2.find)
V.update(RAW=c(rr["duplicate_audit"]["raw_files"]), EXACT=c(rr["duplicate_audit"]["exact_duplicate_files_removed"]),
         UNIQUE=c(len(man)), PHOTOS=c(man.family.nunique()), GROUPS=c(groups.nunique()), XPAIRS=c(len(x0)),
         KTRAIN=c((man.source_split == "train").sum()), KTEST=c(len(te)),
         KFAM_BOTH=c((famsrc == "test+train").sum()), KFAM_TRAIN=c((famsrc == "train").sum()),
         TRUEPAIRS=c(true_pairs), PH_EASY=c(easy), PH_ROT=c(rot), PH_COMPMIN=int(comps.min()), PH_COMPMAX=int(comps.max()),
         PHREC6=pc(pairs[(pairs.hamming <= 6)].same.sum() / true_pairs, 1),
         PH6CROSS=c(pairs[pairs.hamming <= 6].cross_label.sum()), RGBA=c((man["mode"] == "RGBA").sum()),
         DOWNLOADED="13 November 2026")
assert tr_ == 0
rows = []
for k, cl in enumerate(CLASSES):
    m = man[man.label == cl]; t = te[te.label == cl]
    rows.append(f"{NAMES[k]} & {c(m.family.nunique())} & {c(len(m))} & {c(len(t))} & {c(t.family.isin(trf).sum())} \\\\")
rows += ["\\midrule", f"Total & {V['PHOTOS']} & {V['UNIQUE']} & {V['KTEST']} & {c(te.family.isin(trf).sum())} \\\\"]
V["TABLE_DATA"] = "\n".join(rows)
det = []
for t in (0, 2, 4, 6):
    q = pairs[pairs.hamming <= t]
    qs = q[~q.cross_label]
    det.append(f"pHash, $\\tau \\leq {t}$ & {c(len(q))} & {c(q.same.sum())} & {pc(q.same.mean(), 1)} & {pc(qs.same.mean(), 1)} & {pc(q.same.sum() / true_pairs, 1)} \\\\")
    if t == 6:
        V["PH6_LINKS"] = c(len(q)); V["PH6_XFAM"] = c((~q.same).sum()); V["PH6_SCPREC"] = pc(qs.same.mean(), 1); V["PH6_PREC"] = pc(q.same.mean(), 1)

# ------------------------------------------------------------ v1 historical single runs
def v1_views(arch):
    pr = add_provenance(pd.read_csv(arch / "S3_heldout_predictions.csv"))
    ok = pr.label == pr.predicted; o = pr.variant == "original"
    return ok.mean(), ok[o].mean(), ok[~o].mean(), 100 * f1_score(pr.label, pr.predicted, average="macro")
def fmt_p(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e"); return f"{m}\\times 10^{{{int(e)}}}"
    return f"{p:.3f}"
prot_rows = []
for name, arch in (("v1.0 image-level split, seed 42 (unit-scaled input)", V1I), ("v1.0 photo-level split, seed 42 (unit-scaled input)", V1P)):
    a, o, d, f = v1_views(arch); prot_rows.append(f"{name} & {pc(a)} & {pc(o)} & {pc(d)} & {f:.2f} \\\\")

# ------------------------------------------------------------ v2 results
def load_run(rid):
    rec = json.loads((V2 / "runs" / f"{rid}.json").read_text())
    ev = {k: pd.read_csv(V2 / "runs" / f"{rid}.{k}.csv") for k in rec["eval"]}
    return rec, ev

if V2 is not None:
    env = json.loads((V2 / "environment_v2.json").read_text())
    gpus = []
    for ef in sorted(V2.glob("environment_v2*.json")):
        e_ = json.loads(ef.read_text())
        gpus += [g.get("device_name", "") for g in e_.get("gpu", [])]
    sizes = sorted({g.split("-")[-1] for g in gpus if g})
    gpu = "NVIDIA A100-SXM4 " + (" and ".join(sizes) if sizes else "")
    req = dict(l.strip().split("==") for l in (V2 / "requirements_lock_v2.txt").read_text().splitlines() if "==" in l)
    V["ENV"] = (f"{gpu} GPU{'s' if ' and ' in gpu else ''} (NVIDIA Corporation, Santa Clara, CA, USA){', as assigned in different sessions,' if ' and ' in gpu else ''} on Google Colab (Google LLC, Mountain View, CA, USA); mixed precision. "
                f"Python {env['python']}; TensorFlow {env['tensorflow']}; Keras {env['keras']}; NumPy {req.get('numpy','')}; pandas {req.get('pandas','')}; "
                f"SciPy {req.get('scipy','')}; Pillow {req.get('pillow','')}; scikit-learn {env.get('scikit_learn','')}; Matplotlib {req.get('matplotlib','')}; kagglehub {req.get('kagglehub','')}")
    V["E3_EPOCHS"] = env["config"]["E3_EPOCHS_FROZEN"]; V["E3FT_EPOCHS"] = env["config"]["E3_EPOCHS_FT"]
    V["DOWNLOADED"] = pd.Timestamp(json.loads((V2 / "audit" / "provenance.json").read_text())["downloaded_utc"][:10]).strftime("%-d %B %Y")
    A = json.loads((V2 / "audit" / "E1_audit.json").read_text())
    cp = pd.read_csv(V2 / "audit" / "corner_padding.csv", index_col=0)
    rotb = cp.loc[[v for v in cp.index if v.startswith("rotated")], "black"].mean()
    V["CORNERS"] = (f"In the four corner regions of the rotated versions, on average {pc(rotb, 0)}\\% of pixels were near-black, against "
                    f"{pc(cp.loc['original', 'black'], 0)}\\% in the original captures, whose corners were mostly near-white ({pc(cp.loc['original', 'white'], 0)}\\%); "
                    f"the translated copies also had black borders ({pc(cp.loc['translation', 'black'], 0)}\\%). These values are consistent with the creator filling the empty areas with black.")
    pcx = A["cross_family_tau0"]["pixel_check"]
    V["PIXCHECK"] = (f"None of the {A['cross_family_tau0']['pairs']} pairs was identical at pixel level; in every pair the two files had different image dimensions, as expected from two separate screen captures of one source image."
                     if set(pcx) == {"different size"} else "Decoded-pixel comparison showed " + ", ".join(f"{v} pair{'s' if v != 1 else ''} {k}" for k, v in pcx.items()) + ".")
    # E2
    if (V2 / "embed" / "E2_embedding.json").exists():
        E = json.loads((V2 / "embed" / "E2_embedding.json").read_text())
        for t in ["0.95", "0.9", "0.85"]:
            r = E["threshold"][t]
            det.append(f"VGG16 cosine $\\geq {t}$ & {c(r['sibling_links'] + r['nonsibling_links'])} & {c(r['sibling_links'])} & n/a & {pc(r['precision'], 1)} & {pc(r['recall'], 1)} \\\\")
        V["E2_AUC"] = f"{E['auc_sibling_vs_same_class_nonsibling']:.3f}"
        V["E2_SUMMARY"] = f"ranked sibling pairs above same-class non-sibling pairs with an area under the ROC curve of {V['E2_AUC']}"
        V["E2_TOP1"] = pc(E['share_top1_is_sibling'], 1); V["E2_TOP8"] = f"{E['mean_siblings_in_top8']:.1f}"
        V["E2_R85"] = pc(E["threshold"]["0.85"]["recall"], 1); V["E2_P85"] = pc(E["threshold"]["0.85"]["precision"], 1)
        V["E2_R95"] = pc(E["threshold"]["0.95"]["recall"], 1); V["E2_P95"] = pc(E["threshold"]["0.95"]["precision"], 1)
        V["FIG_EMBED"] = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=12cm]{Figure2_embedding.png}\n"
                          "\\caption{Cosine similarity of ImageNet VGG16 embeddings for sibling pairs and for a random sample of non-sibling pairs of the same class.\\label{fig:embed}}\n\\end{figure}")
    # E3
    def e3_summary(tag):
        ids = sorted({p.name.rsplit("_", 1)[0] for p in (V2 / "runs").glob(f"{tag}_s*_f*_clean.json")})
        res = []
        for pid in ids:
            rc, ec = load_run(pid + "_clean"); rx, ex = load_run(pid + "_exposed")
            out = {"pair": pid, "split": pid.split("_f")[1], "seed": pid.split("_s")[1].split("_")[0], "updates_equal": rc["updates"] == rx["updates"]}
            for ep in ("orig", "held"):
                a = ec[ep].sort_values("row"); b = ex[ep].sort_values("row"); assert (a.row.values == b.row.values).all()
                ca = (a.label == a.pred).to_numpy(); cb = (b.label == b.pred).to_numpy()
                out[f"{ep}_clean"] = ca.mean(); out[f"{ep}_exposed"] = cb.mean(); out[f"{ep}_diff"] = 100 * (cb.mean() - ca.mean())
                mc = mcnemar_exact(cb, ca); out[f"{ep}_p"] = mc["p"]; out[f"{ep}_n"] = len(ca)
                out[f"{ep}_ci"] = paired_cluster_bootstrap(pd.DataFrame({"cluster": a.group.values, "a": cb.astype(int), "b": ca.astype(int)}))
            res.append(out)
        return pd.DataFrame(res)
    tab = []; figdata = {}
    for tag, nm in (("E3frozen", "Frozen VGG16"), ("E3ft", "Fine-tuned VGG16")):
        d = e3_summary(tag)
        if d.empty: continue
        d.to_csv(OUTD / f"{tag}_pairs_summary.csv", index=False); figdata[nm] = d
        for ep, en in (("orig", "original captures"), ("held", "withheld derived version")):
            split_means = d.groupby("split")[f"{ep}_diff"].mean().to_numpy()
            ci = t_interval(split_means) if len(split_means) > 1 else [np.nan, np.nan]
            tab.append(f"{nm}, {en} & {pc(d[f'{ep}_clean'].mean())} & {pc(d[f'{ep}_exposed'].mean())} & {d[f'{ep}_diff'].mean():+.2f} & {ci[0]:+.2f} to {ci[1]:+.2f} \\\\")
            key = "E3" if tag == "E3frozen" else "E3FT"
            if ep == "orig":
                V[f"{key}_DIFF"] = f"{d[f'{ep}_diff'].mean():+.2f}"; V[f"{key}_CI"] = f"{ci[0]:+.2f} to {ci[1]:+.2f}"
                V[f"{key}_NPAIRS"] = len(d); V[f"{key}_CLEAN"] = pc(d[f'{ep}_clean'].mean()); V[f"{key}_EXP"] = pc(d[f'{ep}_exposed'].mean())
                V[f"{key}_NSIG"] = int((d[f"{ep}_p"] < 0.05).sum()); V[f"{key}_NPOS"] = int((d[f"{ep}_diff"] > 0).sum())
            else:
                V[f"{key}_HDIFF"] = f"{d[f'{ep}_diff'].mean():+.2f}"; V[f"{key}_HCI"] = f"{ci[0]:+.2f} to {ci[1]:+.2f}"
    for tag, key in (("E3frozen", "E3"), ("E3ft", "E3FT")):
        for ep, suf in (("orig", ""), ("held", "H")):
            errc = erre = n = 0; ps = []; disc = []
            for sd in sorted({p.name.split("_s")[1].split("_")[0] for p in (V2 / "runs").glob(f"{tag}_s*_f*_clean.json")}):
                cc, ce = [], []
                for f in sorted((V2 / "runs").glob(f"{tag}_s{sd}_f*_clean.json")):
                    pid = f.name.rsplit("_", 1)[0]
                    a = load_run(pid + "_clean")[1][ep].sort_values("row"); b = load_run(pid + "_exposed")[1][ep].sort_values("row")
                    cc.append((a.label == a.pred).to_numpy()); ce.append((b.label == b.pred).to_numpy())
                cc = np.concatenate(cc); ce = np.concatenate(ce)
                errc += int((~cc).sum()); erre += int((~ce).sum()); n += len(cc)
                ps.append(mcnemar_exact(ce, cc)["p"]); disc.append((int((cc & ~ce).sum()), int((~cc & ce).sum())))
            if n:
                V[f"{key}{suf}_ERRC"] = errc; V[f"{key}{suf}_ERRE"] = erre; V[f"{key}{suf}_NEVAL"] = c(n)
                V[f"{key}{suf}_ERRCUT"] = f"{100 * (1 - erre / errc):.0f}"
                V[f"{key}{suf}_PMAX"] = fmt_p(max(ps)); V[f"{key}{suf}_DISC"] = disc
    V["TABLE_E3"] = "\n".join(tab)
    if "E3_DIFF" in V:
        dfz = V["E3_DISC"]; dft = V.get("E3FT_DISC", [])
        vd = {}
        for tag in ("E3frozen", "E3ft"):
            ds = []
            for f in (V2 / "runs").glob(f"{tag}_s*_f*_clean.json"):
                rc = json.loads(f.read_text())["eval"]["val"]; rx = json.loads(f.with_name(f.name.replace("_clean", "_exposed")).read_text())["eval"]["val"]
                ds.append(100 * (rx["correct"] / rx["n"] - rc["correct"] / rc["n"]))
            vd[tag] = np.mean(ds) if ds else np.nan
        V["FIG_E3"] = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=13cm]{Figure3_matched.png}\n"
                       "\\caption{Exposed-minus-clean accuracy difference for every fit pair of the matched study (S, seed; F, split), on the original captures and on the withheld derived versions. Vertical bars are 95\\% percentile bootstrap intervals obtained by resampling test capture groups within each fitted pair (5000 draws); they are conditional on the fitted models and differ from the fold-based intervals of Table~\\ref{tab:e3}.\\label{fig:e3}}\n\\end{figure}")
    # E4 / E5
    def view(df):
        ok = df.label == df.pred; o = df.variant == "original"
        return ok.mean(), ok[o].mean(), ok[~o].mean(), f1_score(df.label, df.pred, average="macro", labels=range(6))
    for proto in ("image", "photo"):
        vals = [view(load_run(f"E4{proto}_r{r}")[1]["test"]) for r in range(10) if (V2 / "runs" / f"E4{proto}_r{r}.json").exists()]
        if vals:
            a = np.array(vals) * 100
            lbl = "Image-level split" if proto == "image" else "Capture-group split"
            prot_rows.append(f"E4 {lbl}, mean $\\pm$ SD over {len(vals)} repeats & {a[:,0].mean():.2f} $\\pm$ {a[:,0].std(ddof=1):.2f} & {a[:,1].mean():.2f} $\\pm$ {a[:,1].std(ddof=1):.2f} & {a[:,2].mean():.2f} $\\pm$ {a[:,2].std(ddof=1):.2f} & {a[:,3].mean():.2f} $\\pm$ {a[:,3].std(ddof=1):.2f} \\\\")
            V[f"E4_{proto.upper()}"] = f"{a[:,0].mean():.2f}"
    for arm in ("image", "photo"):
        if (V2 / "runs" / f"E5v1_{arm}.json").exists():
            a, o, d, f = view(load_run(f"E5v1_{arm}")[1]["test"])
            prot_rows.append(f"E5 v1.0 {arm}-level memberships, documented input and v2 pipeline & {pc(a)} & {pc(o)} & {pc(d)} & {100 * f:.2f} \\\\")
    e5 = {arm: view(load_run(f"E5v1_{arm}")[1]["test"])[0] for arm in ("image", "photo") if (V2 / "runs" / f"E5v1_{arm}.json").exists()}
    if len(e5) == 2:
        v1i = v1_views(V1I)[0]; v1p = v1_views(V1P)[0]
        V.update(V1I_ACC=pc(v1i), V1P_ACC=pc(v1p), E5I_ACC=pc(e5["image"]), E5P_ACC=pc(e5["photo"]),
                 GAP_V1=f"{100 * (v1i - v1p):.2f}", GAP_E5=f"{100 * (e5['image'] - e5['photo']):.2f}")
    if "E4_IMAGE" in V:
        V["GAP_E4"] = f"{float(V['E4_IMAGE']) - float(V['E4_PHOTO']):.2f}"
    # review additions: class-wise and version-wise errors, discordance, validation, curves, topology
    import extra_v2
    X = extra_v2.compute(V2, OUTD)
    V.update({k: v for k, v in X.items() if not k.startswith("_") and not isinstance(v, list)})
    def mtxt(seeds):
        parts = [f"{b} against {a} for seed {sd} ($p = {extra_v2.fmt_p(p)}$)" for sd, b, a, p in seeds]
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    V["MCN_FZ"] = mtxt(X["E3_SEEDS"])
    if "E3FT_SEEDS" in X: V["MCN_FT"] = mtxt(X["E3FT_SEEDS"])
    V["CLS_TOP1"], V["CLS_TOP2"] = [z.lower() for z in X["CLS_TOP"]]
    V["CLS_ZERO"] = " and ".join(z.lower() for z in X["CLS_ZERO"]) or "none"
    V["E3FT_SEEDTXT"] = "one seed" if X.get("E3FT_NSEEDS", 1) == 1 else "two seeds"
    V["E3FT_SEEDS_N"] = X.get("E3FT_NSEEDS", 1)
    vt = X["_VT"]
    up_fz = [extra_v2.VNAME[v].lower() for v, (a, b) in vt["E3frozen"].items() if b > a]
    up_ft = [extra_v2.VNAME[v].lower() for v, (a, b) in vt.get("E3ft", {}).items() if b > a]
    V["VER_FZ"] = ("fewer errors in every version type" if not up_fz else "fewer errors in every version type except the " + ", ".join(up_fz))
    V["VER_FT"] = ("fewer errors in every version type" if not up_ft else "fewer errors in every version type except the " + ", ".join(up_ft).replace("\\textdegree", "\\textdegree{}"))
    if up_ft:
        v = [k for k, (a, b) in vt["E3ft"].items() if b > a][0]
        V["VER_FT_EXC"] = f" ({vt['E3ft'][v][1]} errors against {vt['E3ft'][v][0]} in the clean arm)"
    else:
        V["VER_FT_EXC"] = ""
    R_ = X["_REG"]
    V["E3_NNEG"] = int((R_.query("backbone == 'frozen'").orig_diff < 0).sum()); V["E3_NTIE"] = int((R_.query("backbone == 'frozen'").orig_diff == 0).sum())
    W = {0: "none", 1: "one", 2: "two", 3: "three"}
    V["E3_NNEG_TXT"] = W.get(V["E3_NNEG"], str(V["E3_NNEG"]))
    V["E3FT_NNEG"] = int((R_.query("backbone == 'fine-tuned'").orig_diff < 0).sum())
    V["CLS_FZ_N"] = c(X["_REG"].query("backbone == 'frozen'").orig_n.sum())
    import shutil
    V["F1_E4I"] = f"{np.mean([X['_F1'][k] for k in X['_F1'] if k.startswith('E4image')]):.2f}"
    V["F1_E4P"] = f"{np.mean([X['_F1'][k] for k in X['_F1'] if k.startswith('E4photo')]):.2f}"
    # figures that need v2
    subprocess.run([sys.executable, str(HERE / "figures_v2.py"), str(V2), str(OUTD / "Figures"), str(OUTD)], check=True)
V["TABLE_DETECT"] = "\n".join(det)
V["TABLE_PROTOCOLS"] = "\n".join(prot_rows)
V.setdefault("ZENODO", "version 2.0 DOI to be added on release")

keys = []
if (HERE / "manuscript_template.tex").exists():
    tpl = (HERE / "manuscript_template.tex").read_text(encoding="utf-8")
    keys = sorted(set(re.findall(r"<<([A-Z0-9_]+)>>", tpl)))
    for k in keys:
        if k not in V:
            V[k] = ("" if k.startswith("FIG_") else
                    ("\\multicolumn{5}{c}{" + pend(k.replace("_", " ") + ": filled from the v2 run") + "} \\\\" if k.startswith("TABLE_") else pend(k.replace("_", " "))))
    out = re.sub(r"<<([A-Z0-9_]+)>>", lambda m: str(V[m.group(1)]), tpl)
    (HERE / "manuscript.tex").write_text(out, encoding="utf-8")
(OUTD / "numbers_used_v2.json").write_text(json.dumps({k: v for k, v in V.items() if not k.startswith(("TABLE", "FIG"))}, indent=1, default=str))
print("pending:", [k for k in keys if "\\pend" in str(V[k])])
