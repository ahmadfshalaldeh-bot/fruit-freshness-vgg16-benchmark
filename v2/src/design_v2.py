"""Pure-Python design and statistics helpers for the v2 study (no TensorFlow needed).

Used by the Colab notebook and by the manuscript build script, so the same code defines
the experiment and analyses its outputs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLASSES = ["freshapples", "rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges"]
NAMES = ["Fresh apple", "Rotten apple", "Fresh banana", "Rotten banana", "Fresh orange", "Rotten orange"]
OP_RE = r"^(rotated_by_\d+|saltandpepper|translation|vertical_flip)_"
VARIANTS = ["original", "rotated_by_15", "rotated_by_30", "rotated_by_45", "rotated_by_60", "rotated_by_75",
            "saltandpepper", "translation", "vertical_flip"]
DERIVED = VARIANTS[1:]


def add_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Add stored-version and filename-derived base-capture family columns."""
    df = df.copy()
    fn = df["relative_path"].str.split("/").str[-1]
    df["variant"] = fn.str.extract(OP_RE)[0].fillna("original")
    df["base_name"] = fn.str.replace(OP_RE, "", regex=True)
    df["family"] = df["label"] + "|" + df["base_name"]
    return df


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def capture_groups(df: pd.DataFrame, tau0_pairs: list[tuple[int, int]]) -> pd.Series:
    """Merge filename families that are linked by any zero-distance pHash pair (prediction-blind rule).
    df must have positional index 0..n-1 and a 'family' column; pairs are positional indices."""
    uf = UnionFind(df["family"].unique())
    fam = df["family"].to_numpy()
    for i, j in tau0_pairs:
        uf.union(fam[i], fam[j])
    return df["family"].map(uf.find)


def stratified_group_folds(groups: pd.Series, labels: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Assign every group to one of n_folds, balancing classes: groups are shuffled within class and dealt round-robin.
    Returns fold id per row. Deterministic for a given seed."""
    g = pd.DataFrame({"group": groups.to_numpy(), "label": labels.to_numpy()})
    glab = g.groupby("group")["label"].agg(lambda s: s.mode().iloc[0])
    rng = np.random.default_rng(seed)
    fold_of = {}
    for lab in sorted(glab.unique()):
        gs = np.array(sorted(glab[glab == lab].index))
        rng.shuffle(gs)
        offset = rng.integers(n_folds)
        for k, gid in enumerate(gs):
            fold_of[gid] = (k + offset) % n_folds
    return g["group"].map(fold_of).to_numpy()


def matched_design(df: pd.DataFrame, fold: np.ndarray, test_fold: int, n_folds: int, seed: int) -> dict:
    """Matched contamination design for one split/replicate.

    * test groups  = fold == test_fold; one evaluation original per test family plus one withheld derived version.
    * validation   = a class-balanced half of the groups in the next fold, protected from all training.
    * donors       = one distinct same-class family per test family, drawn from the remaining pool.
    * base         = all other pool families (all nine versions).
    * clean arm    = base + donor bundles;  exposed arm = base + exposure bundles of the test families.
      A bundle is the seven derived versions other than the test family's withheld version type, so both arms have
      identical training-file counts, class counts and stored-version composition.
    Returns positional index arrays and a per-test-family table."""
    rng = np.random.default_rng(seed)
    n = len(df)
    pos = np.arange(n)
    is_test = fold == test_fold
    nxt = (test_fold + 1) % n_folds
    # protected validation: class-balanced half of next fold's groups
    nf = df[fold == nxt]
    vgroups = []
    for lab, d in nf.groupby("label"):
        gs = np.array(sorted(d["group"].unique()))
        rng.shuffle(gs)
        vgroups += list(gs[: len(gs) // 2])
    is_val = df["group"].isin(set(vgroups)).to_numpy() & (fold == nxt)
    is_pool = ~is_test & ~is_val
    fam_lab = df.groupby("family")["label"].first()
    test_fams = sorted(df.loc[is_test, "family"].unique())
    pool_fams = sorted(df.loc[is_pool, "family"].unique())
    # withheld derived version: balanced cycle over the eight derived types within class
    withheld = {}
    for lab in CLASSES:
        fs = [f for f in test_fams if fam_lab[f] == lab]
        rng.shuffle(fs)
        off = rng.integers(8)
        for k, f in enumerate(fs):
            withheld[f] = DERIVED[(k + off) % 8]
    # donors: distinct same-class pool families
    donor = {}
    for lab in CLASSES:
        fs = [f for f in test_fams if fam_lab[f] == lab]
        cand = [f for f in pool_fams if fam_lab[f] == lab]
        if len(cand) < len(fs):
            raise ValueError(f"not enough donor families for {lab}: {len(cand)} < {len(fs)}")
        pick = rng.choice(len(cand), size=len(fs), replace=False)
        for f, k in zip(fs, pick):
            donor[f] = cand[k]
    donors = set(donor.values())
    by_fam = {f: d for f, d in df.groupby("family")}
    base = df.index[is_pool & ~df["family"].isin(donors).to_numpy()].to_numpy()
    clean_add, exposed_add, eval_orig, eval_held = [], [], [], []
    rows = []
    for f in test_fams:
        w = withheld[f]
        bundle_types = [v for v in DERIVED if v != w]
        tf_ = by_fam[f]; dn = by_fam[donor[f]]
        exp_idx = tf_.index[tf_["variant"].isin(bundle_types)].to_numpy()
        don_idx = dn.index[dn["variant"].isin(bundle_types)].to_numpy()
        if len(exp_idx) != 7 or len(don_idx) != 7:
            raise ValueError(f"family {f} or donor lacks a complete 7-version bundle")
        exposed_add += list(exp_idx); clean_add += list(don_idx)
        o = tf_.index[tf_["variant"] == "original"].to_numpy(); h = tf_.index[tf_["variant"] == w].to_numpy()
        eval_orig += list(o); eval_held += list(h)
        rows.append({"test_family": f, "label": fam_lab[f], "donor_family": donor[f], "withheld_version": w,
                     "eval_original": int(o[0]), "eval_withheld": int(h[0])})
    val = df.index[is_val].to_numpy()
    out = {"base": base, "clean_train": np.concatenate([base, clean_add]), "exposed_train": np.concatenate([base, exposed_add]),
           "val": val, "eval_original": np.array(eval_orig), "eval_withheld": np.array(eval_held),
           "pairs": pd.DataFrame(rows)}
    # integrity checks
    ct, et = set(out["clean_train"]), set(out["exposed_train"])
    ev = set(out["eval_original"]) | set(out["eval_withheld"])
    assert not (ct & ev) and not (et & ev), "an evaluation file entered training"
    assert not (ct & set(val)) and not (et & set(val)), "validation overlaps training"
    assert len(ct) == len(et) == len(out["clean_train"]) == len(out["exposed_train"])
    assert (df.loc[list(ct), "label"].value_counts().sort_index() == df.loc[list(et), "label"].value_counts().sort_index()).all()
    assert (df.loc[list(ct), "variant"].value_counts().sort_index() == df.loc[list(et), "variant"].value_counts().sort_index()).all()
    test_groups = set(df.loc[is_test, "group"])
    assert not (set(df.loc[list(ct), "group"]) & test_groups), "clean arm contains a test capture group"
    assert not (set(df.loc[val, "group"]) & set(df.loc[list(et), "group"]) & test_groups)
    return out


# ---------------------------------------------------------------- statistics
def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest
    b = int((a_correct & ~b_correct).sum()); c = int((~a_correct & b_correct).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_only": b, "b_only": c, "p": float(p)}


def paired_cluster_bootstrap(diff_by_cluster: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> list[float]:
    """diff_by_cluster: columns cluster, a (0/1), b (0/1). Resample clusters; return 95% CI of mean(a)-mean(b) in pp."""
    rng = np.random.default_rng(seed)
    g = diff_by_cluster.groupby("cluster")[["a", "b"]].agg(["sum", "count"])
    sa, n = g[("a", "sum")].to_numpy(), g[("a", "count")].to_numpy(); sb = g[("b", "sum")].to_numpy()
    k = len(sa); vals = np.empty(n_boot)
    for t in range(n_boot):
        ii = rng.integers(0, k, k)
        vals[t] = 100 * (sa[ii].sum() - sb[ii].sum()) / n[ii].sum()
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def t_interval(x: np.ndarray, conf: float = 0.95) -> list[float]:
    from scipy.stats import t
    x = np.asarray(x, float); m = x.mean(); s = x.std(ddof=1) / np.sqrt(len(x))
    h = t.ppf(0.5 + conf / 2, len(x) - 1) * s
    return [float(m - h), float(m + h)]


# ---------------------------------------------------------------- v1 split reconstruction
def reconstruct_v1_splits(work_sorted: pd.DataFrame) -> dict:
    """Rebuild the two seed-42 v1 splits. work_sorted must be the unique images sorted by relative_path
    (the v1 order), with 'label' and 'family'."""
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold
    n = len(work_sorted); idx = np.arange(n)
    y = work_sorted["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    n_test = round(0.15 * n); n_val = round(0.15 * n)
    trval, test = train_test_split(idx, test_size=n_test, stratify=y, random_state=42)
    train, val = train_test_split(trval, test_size=n_val, stratify=y[trval], random_state=42)
    img = np.empty(n, dtype=object); img[train], img[val], img[test] = "train", "validation", "test"
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    folds = [te for _, te in sgkf.split(idx, y, groups=work_sorted["family"])]
    ph = np.empty(n, dtype=object)
    ph[np.concatenate(folds[:3])] = "test"; ph[np.concatenate(folds[3:6])] = "validation"; ph[np.concatenate(folds[6:])] = "train"
    return {"image": img, "photo": ph}
