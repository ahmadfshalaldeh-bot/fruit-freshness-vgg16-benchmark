# %% [markdown]
# # One Photograph, Nine Files — v2 experiments (provenance audit + matched contamination study)
#
# **Run on Google Colab:** Runtime → Change runtime type → **A100 GPU + High RAM** → Runtime → **Run all**.
# Results are written to Google Drive after every fit, so if Colab disconnects just press **Run all** again:
# finished fits are skipped automatically.
#
# Experiments (switch on/off in cell 0):
# * **E1** provenance audit (no training): Kaggle's own train/test folders, hash-link structure, corner padding, repeated captures.
# * **E2** sibling detection with VGG16 embeddings (no training).
# * **E3** matched contamination study (primary): identical test originals, clean vs exposed training, 5 splits × 2 seeds.
# * **E3-FT** the same design with a fully fine-tuned VGG16 (5 splits × 1 seed).
# * **E4** repeated protocol comparison: image-level vs photo-level split, 5 repeats each.
# * **E5** the two seed-42 splits of v1 re-run with the documented VGG16 preprocessing.
#
# Approximate A100 time: E1–E2 20 min, E3 1.5–2 h, E3-FT about 1 h, E4 about 1 h, E5 20 min (3.5–4.5 h in total).

# %% 0. Configuration
import os, sys
from pathlib import Path

RUN_E1, RUN_E2, RUN_E3, RUN_E3_FT, RUN_E4, RUN_E5 = True, True, True, True, True, True
USE_DRIVE = True                     # keep results in Google Drive so a disconnect loses nothing
PREPROCESS = "vgg_caffe"             # documented VGG16 input: RGB->BGR, ImageNet mean subtraction, no /255
MIXED_PRECISION = True
IMG_SIZE, BATCH = 224, 32
N_FOLDS = 5
E3_SEEDS = [1, 2]                    # replicate seeds per split (frozen backbone)
E3_FT_SEEDS = [1, 2]
E3_EPOCHS_FROZEN = 30                # fixed, predeclared update budget (no early stopping, final weights evaluated)
E3_EPOCHS_FT = 12
LR_FROZEN, LR_FT = 1e-4, 1e-5
E4_REPEATS = 3                       # repeated protocol comparison (early stopping as in v1)
E4_MAX_EPOCHS, E4_PATIENCE = 100, 10
PHASH_MAX = 6
KAGGLE_DATASET = "sriramr/fruits-fresh-and-rotten-for-classification"
V1_MANIFEST_URLS = {   # archived v1 split assignments (exact memberships for E5)
    "image": "https://raw.githubusercontent.com/ahmadfshalaldeh-bot/fruit-freshness-vgg16-benchmark/v1.0/results/image_split/S1_manifest_with_split.csv",
    "photo": "https://raw.githubusercontent.com/ahmadfshalaldeh-bot/fruit-freshness-vgg16-benchmark/v1.0/results/photo_split/S1_manifest_with_split.csv",
}

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive/fruit_v2_results")
else:
    OUT = Path("/content/fruit_v2_results") if IN_COLAB else Path("fruit_v2_results")
for sub in ["audit", "embed", "runs", "designs"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
print("Results folder:", OUT)

# %% 1. Imports, seeds and the shared design module
import json, hashlib, platform, subprocess, datetime, random, zipfile, math, time
import numpy as np, pandas as pd
from PIL import Image
random.seed(0); np.random.seed(0)
sys.path.insert(0, ".")   # design_v2.py is written by the cell above
from design_v2 import (CLASSES, NAMES, VARIANTS, DERIVED, add_provenance, capture_groups,
                       stratified_group_folds, matched_design)
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

# %% 2. Download and provenance record
try:
    import kagglehub
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True); import kagglehub
DATA_ROOT = Path(kagglehub.dataset_download(KAGGLE_DATASET))
prov = {"source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}", "kagglehub_path": str(DATA_ROOT),
        "downloaded_utc": utc(), "kagglehub_version": getattr(kagglehub, "__version__", "unknown")}
(OUT / "audit" / "provenance.json").write_text(json.dumps(prov, indent=2)); print(prov)

# %% 3. Inventory, checksums and perceptual hashes (cached)
from scipy.fft import dctn
INV = OUT / "audit" / "inventory.csv"; PH = OUT / "audit" / "phash_d4.npy"
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def phash_d4(img):
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64); out = []
    for flip in (False, True):
        a = np.fliplr(g) if flip else g
        for k in range(4):
            d = dctn(np.rot90(a, k), norm="ortho")[:8, :8]
            out.append(np.packbits((d > np.median(d)).flatten()).view(">u8")[0])
    return np.array(out, dtype=np.uint64)
if INV.exists() and PH.exists():
    inv = pd.read_csv(INV); PHASH = np.load(PH)
else:
    lookup = {c.lower(): c for c in CLASSES}; rows = []
    for p in sorted(DATA_ROOT.rglob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.parent.name.lower() in lookup:
            parts = [s.lower() for s in p.relative_to(DATA_ROOT).parts]
            rows.append({"path": str(p), "relative_path": p.relative_to(DATA_ROOT).as_posix(), "label": lookup[p.parent.name.lower()],
                         "source_split": "train" if "train" in parts else ("test" if "test" in parts else "unspecified")})
    inv = pd.DataFrame(rows); hs = []; sh = []; meta = []
    for i, p in enumerate(inv.path):
        sh.append(sha256_file(p))
        with Image.open(p) as im:
            im.load(); meta.append((im.width, im.height, im.mode)); hs.append(phash_d4(im.convert("RGB")))
        if i % 3000 == 0: print("hashed", i)
    inv["sha256"] = sh; inv[["width", "height", "mode"]] = pd.DataFrame(meta)
    PHASH = np.stack(hs); inv.to_csv(INV, index=False); np.save(PH, PHASH)
inv = add_provenance(inv)
print(len(inv), "files")

# %% 4. Analysis set (exact copies removed) and capture groups
inv["phash_row"] = np.arange(len(inv))
work = inv.sort_values("relative_path").drop_duplicates("sha256", keep="first").reset_index(drop=True)
H = PHASH[work.phash_row.to_numpy()]
def popcount(x):
    return np.bitwise_count(x) if hasattr(np, "bitwise_count") else np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)
pairs = []
for s in range(0, len(work), 128):
    d = popcount(H[s:s + 128][:, None, :] ^ H[:, 0][None, :, None]).min(-1)
    ii, jj = np.nonzero(d <= PHASH_MAX); ii += s; keep = jj > ii
    pairs += [(int(i), int(j), int(d[i - s, j])) for i, j in zip(ii[keep], jj[keep])]
pairs = pd.DataFrame(pairs, columns=["i", "j", "hamming"])
work["group"] = capture_groups(work, list(zip(pairs.loc[pairs.hamming == 0, "i"], pairs.loc[pairs.hamming == 0, "j"])))
work["label_idx"] = work.label.map({c: k for k, c in enumerate(CLASSES)})
work.drop(columns=["path"]).to_csv(OUT / "audit" / "manifest_v2.csv", index=False)
pairs.to_csv(OUT / "audit" / "phash_pairs.csv", index=False)
print(len(work), "unique images |", work.family.nunique(), "families |", work.group.nunique(), "capture groups")

# %% 5. E1 — provenance audit outputs
if RUN_E1:
    A = {}
    fam_src = work.groupby("family").source_split.agg(lambda s: "+".join(sorted(set(s))))
    te = work[work.source_split == "test"]; trf = set(work[work.source_split == "train"].family)
    A["kaggle_native"] = {"families_by_folder": fam_src.value_counts().to_dict(), "official_test_images": len(te),
                          "official_test_with_sibling_in_official_train": int(te.family.isin(trf).sum()),
                          "per_class": {c: [int(d.family.isin(trf).sum()), len(d)] for c, d in te.groupby("label")}}
    fam = work.family.to_numpy(); var = work.variant.to_numpy()
    pairs["same_family"] = fam[pairs.i] == fam[pairs.j]
    pairs["version_pair"] = ["|".join(sorted(x)) for x in zip(var[pairs.i], var[pairs.j])]
    pairs["cross_label"] = work.label.to_numpy()[pairs.i] != work.label.to_numpy()[pairs.j]
    true_pairs = int(sum(k * (k - 1) // 2 for k in work.groupby("family").size()))
    A["phash"] = {"true_sibling_pairs": true_pairs, "by_threshold": {}}
    for t in (0, 2, 4, 6):
        q = pairs[pairs.hamming <= t]
        A["phash"]["by_threshold"][t] = {"linked": len(q), "sibling": int(q.same_family.sum()), "cross_label": int(q.cross_label.sum()),
                                         "true_links_by_version_pair": q[q.same_family].version_pair.value_counts().to_dict()}
    # residual repeated captures: zero-distance links between different families
    x = pairs[(pairs.hamming == 0) & ~pairs.same_family].copy()
    eq = []
    for i, j in zip(x.i, x.j):
        a = np.asarray(Image.open(work.path[i]).convert("RGB")); b = np.asarray(Image.open(work.path[j]).convert("RGB"))
        if a.shape == b.shape:
            eq.append(("identical" if np.array_equal(a, b) else "same size, different pixels", float(np.abs(a.astype(int) - b).mean())))
        else:
            eq.append(("different size", float("nan")))
    x["pixel_check"], x["mean_abs_diff"] = [e[0] for e in eq], [e[1] for e in eq]
    x.assign(file_i=work.relative_path[x.i].values, file_j=work.relative_path[x.j].values).to_csv(OUT / "audit" / "cross_family_tau0_pairs.csv", index=False)
    A["cross_family_tau0"] = {"pairs": len(x), "pixel_check": x.pixel_check.value_counts().to_dict(),
                              "families": int(work.family.nunique()), "capture_groups": int(work.group.nunique())}
    # corner padding of the creator's stored versions (fraction of near-black / near-white / transparent corner pixels)
    rows = []
    rng = np.random.default_rng(0)
    for v, d in work.groupby("variant"):
        for p in d.path.sample(min(150, len(d)), random_state=0):
            im = Image.open(p); arr = np.asarray(im.convert("RGBA")); h, w = arr.shape[:2]; c = max(2, int(0.08 * min(h, w)))
            corners = np.concatenate([arr[:c, :c].reshape(-1, 4), arr[:c, -c:].reshape(-1, 4), arr[-c:, :c].reshape(-1, 4), arr[-c:, -c:].reshape(-1, 4)])
            rgb = corners[:, :3].astype(int)
            rows.append({"variant": v, "black": float((rgb.max(1) <= 10).mean()), "white": float((rgb.min(1) >= 245).mean()),
                         "transparent": float((corners[:, 3] < 10).mean()), "mode": im.mode})
    cs = pd.DataFrame(rows).groupby("variant")[["black", "white", "transparent"]].mean().reindex(VARIANTS)
    cs.to_csv(OUT / "audit" / "corner_padding.csv"); A["corner_padding"] = cs.round(4).to_dict(orient="index")
    (OUT / "audit" / "E1_audit.json").write_text(json.dumps(A, indent=2, default=str))
    print(json.dumps({k: A[k] for k in ["kaggle_native", "cross_family_tau0"]}, indent=1, default=str)); print(cs)

# %% 6. TensorFlow, image cache and preprocessing
import tensorflow as tf
if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(0)  # determinism needs a global seed; each fit re-seeds itself
except Exception as e:
    print("determinism unavailable:", e)
print("TF", tf.__version__, tf.config.list_physical_devices("GPU"))
CACHE = Path("/content/img_cache_224.npy") if IN_COLAB else Path("img_cache_224.npy")
if CACHE.exists():
    IMGS = np.load(CACHE)
else:
    IMGS = np.empty((len(work), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for k, p in enumerate(work.path):
        im = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        IMGS[k] = tf.cast(tf.clip_by_value(tf.round(tf.image.resize(im, (IMG_SIZE, IMG_SIZE), method="bilinear")), 0, 255), tf.uint8).numpy()
        if k % 3000 == 0: print("decoded", k)
    np.save(CACHE, IMGS)
Y = work.label_idx.to_numpy(np.int32)
MEAN_BGR = tf.constant([103.939, 116.779, 123.68], tf.float32)
def to_model_input(x):            # x: float32 RGB in [0, 255]
    if PREPROCESS == "vgg_caffe":
        return x[..., ::-1] - MEAN_BGR
    return x / 255.0
DEG = math.pi / 180.0
def _mat(*v):                     # nine [B] tensors -> [B, 3, 3]
    return tf.reshape(tf.stack(v, axis=-1), [-1, 3, 3])
def augment_batch(x, sd):         # x: [B,H,W,3] float32 in [0,255]; sd: int64 [2] seed. Stateless, so tf.data can run it in parallel
    B = tf.shape(x)[0]; S = float(IMG_SIZE); one, zero = tf.ones([B]), tf.zeros([B])
    ks = tf.random.experimental.stateless_split(sd, num=9)
    U = lambda k, shape, lo, hi: tf.random.stateless_uniform(shape, seed=ks[k], minval=lo, maxval=hi)
    x = tf.where(U(0, [B, 1, 1, 1], 0.0, 1.0) < 0.5, x[:, :, ::-1, :], x)
    x = tf.where(U(1, [B, 1, 1, 1], 0.0, 1.0) < 0.5, x[:, ::-1, :, :], x)
    th = U(2, [B], -20.0, 20.0) * DEG; sh = U(3, [B], -0.1, 0.1) * DEG
    zx, zy = U(4, [B], 0.8, 1.2), U(5, [B], 0.8, 1.2)
    tx, ty = U(6, [B], -0.1, 0.1) * S, U(7, [B], -0.1, 0.1) * S
    c, s = tf.cos(th), tf.sin(th); ctr = (S - 1.0) / 2.0
    Cf = _mat(one, zero, one * ctr, zero, one, one * ctr, zero, zero, one)
    Cb = _mat(one, zero, -one * ctr, zero, one, -one * ctr, zero, zero, one)
    T = _mat(one, zero, tx, zero, one, ty, zero, zero, one)
    R = _mat(c, -s, zero, s, c, zero, zero, zero, one)
    SH = _mat(one, -tf.sin(sh), zero, zero, tf.cos(sh), zero, zero, zero, one)
    Z = _mat(zx, zero, zero, zero, zy, zero, zero, zero, one)
    M = Cf @ T @ R @ SH @ Z @ Cb
    M = M / M[:, 2:3, 2:3]
    tv = tf.stack([M[:, 0, 0], M[:, 0, 1], M[:, 0, 2], M[:, 1, 0], M[:, 1, 1], M[:, 1, 2], M[:, 2, 0], M[:, 2, 1]], axis=-1)
    x = tf.raw_ops.ImageProjectiveTransformV3(images=x, transforms=tv, output_shape=tf.constant([IMG_SIZE, IMG_SIZE]),
                                              fill_value=0.0, interpolation="BILINEAR", fill_mode="NEAREST")
    return tf.clip_by_value(x * U(8, [B, 1, 1, 1], 0.8, 1.2), 0.0, 255.0)
Y1H = np.eye(6, dtype=np.float32)[Y]
def _fetch_batch(b):              # one numpy gather per batch (cheap; runs serially under op determinism)
    x = tf.numpy_function(lambda j: IMGS[j], [b], tf.uint8); x.set_shape((None, IMG_SIZE, IMG_SIZE, 3))
    y = tf.numpy_function(lambda j: Y1H[j], [b], tf.float32); y.set_shape((None, 6))
    return x, y
def make_ds(idx, training, seed=0, raw=False):
    """raw=True: float32 RGB in [0,255]; augmentation and VGG preprocessing then run on the GPU inside the model.
    Training streams are shuffled and repeated; fit() is given steps_per_epoch=ceil(n/BATCH)."""
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(idx, np.int64))
    if training:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).map(_fetch_batch)
    def _raw(x, y): return tf.cast(x, tf.float32), y
    def _pre(x, y): return to_model_input(tf.cast(x, tf.float32)), y
    f = _raw if raw else _pre
    ds = ds.map(f, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

_ds = make_ds(np.arange(min(4096, len(work))), True, raw=True)
for xb, yb in _ds.take(4): pass
_t = time.time(); _n = 0
for xb, yb in _ds.take(64):
    _n += 1
print("pipeline check:", tuple(xb.shape), tuple(yb.shape), "range", float(tf.reduce_min(xb)), "to", float(tf.reduce_max(xb)),
      f"| {(time.time() - _t) / _n * 1000:.0f} ms per batch | CPUs: {os.cpu_count()}")

# %% 7. Models and the generic fit/evaluate routine
class VGGPreprocess(tf.keras.layers.Layer):
    """RGB [0,255] -> VGG16 'caffe' input (BGR, ImageNet mean subtracted), or unit scaling if PREPROCESS != 'vgg_caffe'."""
    def call(self, x):
        if PREPROCESS == "vgg_caffe":
            return tf.keras.ops.flip(x, axis=-1) - tf.keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype="float32")
        return x / 255.0

class RandomBrightnessScale(tf.keras.layers.Layer):
    """Multiplicative brightness in [lo, hi] per image (as Keras ImageDataGenerator brightness_range), training only."""
    def __init__(self, lo=0.8, hi=1.2, **kw):
        super().__init__(**kw); self.lo, self.hi = lo, hi; self.sg = tf.keras.random.SeedGenerator()
    def call(self, x, training=None):
        if not training:
            return x
        f = tf.keras.random.uniform((tf.keras.ops.shape(x)[0], 1, 1, 1), self.lo, self.hi, seed=self.sg)
        return tf.keras.ops.clip(x * f, 0.0, 255.0)

def build_model(fine_tune):
    """Augmentation runs on the GPU and only during training (flips, +-20 deg rotation, independent x/y zoom 0.8-1.2,
    +-10% shifts, brightness 0.8-1.2, nearest fill), matching the v1 ImageDataGenerator policy except its 0.1-degree shear."""
    L = tf.keras.layers; f32 = dict(dtype="float32")
    base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = bool(fine_tune)
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = L.RandomFlip("horizontal_and_vertical", **f32)(inp)
    x = L.RandomRotation(20 / 360, fill_mode="nearest", **f32)(x)
    x = L.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest", **f32)(x)
    x = L.RandomTranslation(0.1, 0.1, fill_mode="nearest", **f32)(x)
    x = RandomBrightnessScale(**f32)(x)
    x = VGGPreprocess(**f32)(x)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(256, activation="relu")(x); x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation="relu")(x); x = L.Dropout(0.5)(x)
    out = L.Dense(6, activation="softmax", dtype="float32")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(LR_FT if fine_tune else LR_FROZEN), loss="categorical_crossentropy",
              metrics=["accuracy"], jit_compile=False)
    return m

def run_fit(run_id, train_idx, val_idx, eval_sets, fine_tune, seed, epochs, early_stop=False, init_from=None):
    """Train once and save history, predictions and a run record. Skips if the record exists."""
    rec_path = OUT / "runs" / f"{run_id}.json"
    if rec_path.exists():
        print("skip", run_id); return json.loads(rec_path.read_text())
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    model = build_model(fine_tune)
    init_path = OUT / "runs" / f"{init_from}.init_gpuaug.weights.h5" if init_from else None
    if init_path is not None and init_path.exists():
        model.load_weights(str(init_path))            # identical initial weights for the paired arm
    elif init_from:
        model.save_weights(str(init_path))
    start_digest = hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes() for w in model.get_weights())).hexdigest()
    tf.keras.utils.set_random_seed(seed)              # same RNG stream for data order and augmentation
    cbs = [tf.keras.callbacks.CSVLogger(str(OUT / "runs" / f"{run_id}.history.csv"))]
    if early_stop:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=E4_PATIENCE, restore_best_weights=True))
    t0 = time.time()
    h = model.fit(make_ds(train_idx, True, seed, raw=True), steps_per_epoch=math.ceil(len(train_idx) / BATCH), validation_data=make_ds(val_idx, False, raw=True), epochs=epochs, callbacks=cbs, verbose=2)
    rec = {"run_id": run_id, "fine_tune": fine_tune, "seed": seed, "epochs_run": len(h.history["loss"]),
           "updates": int(len(h.history["loss"]) * math.ceil(len(train_idx) / BATCH)), "n_train": int(len(train_idx)),
           "n_val": int(len(val_idx)), "minutes": round((time.time() - t0) / 60, 2), "preprocess": PREPROCESS,
           "finished_utc": utc(), "start_weights_sha256": start_digest,
           "optimizer_iterations": int(model.optimizer.iterations.numpy()), "eval": {}}
    if early_stop:
        rec["best_epoch"] = int(np.argmin(h.history["val_loss"]) + 1)
    for name, idx in eval_sets.items():
        pr = model.predict(make_ds(idx, False, raw=True), verbose=0)
        df = pd.DataFrame({"row": idx, "relative_path": work.relative_path.to_numpy()[idx], "family": work.family.to_numpy()[idx],
                           "group": work.group.to_numpy()[idx], "variant": work.variant.to_numpy()[idx], "label": Y[idx], "pred": pr.argmax(1)})
        for k in range(6): df[f"p{k}"] = pr[:, k].round(6)
        df.to_csv(OUT / "runs" / f"{run_id}.{name}.csv", index=False)
        rec["eval"][name] = {"n": len(df), "correct": int((df.label == df.pred).sum())}
    rec_path.write_text(json.dumps(rec, indent=2)); print(run_id, rec["eval"], rec["minutes"], "min")
    return rec

tf.keras.utils.set_random_seed(0)
_m = build_model(False); _d = make_ds(np.arange(min(4096, len(work))), True, raw=True)
_m.fit(_d, steps_per_epoch=5, epochs=1, verbose=0)                     # build and warm up
_t = time.time(); _m.fit(_d, steps_per_epoch=40, epochs=1, verbose=0)
print(f"training speed check: {(time.time() - _t) / 40 * 1000:.0f} ms per step (frozen VGG16, batch {BATCH})")
del _m, _d; tf.keras.backend.clear_session()

# %% 8. E2 — sibling detection with ImageNet VGG16 embeddings (no training)
if RUN_E2 and not (OUT / "embed" / "E2_embedding.json").exists():
    tf.keras.utils.set_random_seed(0)
    enc = tf.keras.Sequential([tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
                               tf.keras.layers.GlobalAveragePooling2D()])
    F = enc.predict(make_ds(np.arange(len(work)), False), verbose=0).astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True); np.save(OUT / "embed" / "vgg16_gap_l2.npy", F)
    fam = work.family.to_numpy(); lab = work.label.to_numpy()
    sims_sib, sims_non = [], []; rng = np.random.default_rng(0); topk_hits = []; top1 = []; res = {}
    thr = [0.80, 0.85, 0.90, 0.95, 0.98]; tp = {t: 0 for t in thr}; fp = {t: 0 for t in thr}
    for c in CLASSES:
        ix = np.where(lab == c)[0]; S = F[ix] @ F[ix].T; same = fam[ix][:, None] == fam[ix][None, :]
        iu = np.triu_indices(len(ix), 1); s = S[iu]; m = same[iu]
        sims_sib.append(s[m]); sims_non.append(rng.choice(s[~m], size=min(200000, (~m).sum()), replace=False))
        for t in thr: tp[t] += int((s[m] >= t).sum()); fp[t] += int((s[~m] >= t).sum())
        np.fill_diagonal(S, -1); top = np.argsort(-S, axis=1)[:, :8]
        topk_hits.append((fam[ix][top] == fam[ix][:, None]).sum(1)); top1.append(fam[ix][top[:, 0]] == fam[ix])
    sib = np.concatenate(sims_sib); non = np.concatenate(sims_non); hits = np.concatenate(topk_hits)
    from sklearn.metrics import roc_auc_score
    total = len(sib)
    res = {"true_sibling_pairs": int(total), "auc_sibling_vs_same_class_nonsibling": float(roc_auc_score(
               np.r_[np.ones(len(sib)), np.zeros(len(non))], np.r_[sib, non])),
           "threshold": {t: {"sibling_links": tp[t], "recall": tp[t] / total, "nonsibling_links": fp[t],
                             "precision": tp[t] / max(1, tp[t] + fp[t])} for t in thr},
           "mean_siblings_in_top8": float(hits.mean()), "share_top1_is_sibling": float(np.concatenate(top1).mean()),
           "sibling_sim_quantiles": np.percentile(sib, [5, 25, 50, 75, 95]).round(4).tolist(),
           "nonsibling_sim_quantiles": np.percentile(non, [5, 25, 50, 75, 95]).round(4).tolist()}
    np.save(OUT / "embed" / "sim_sibling.npy", sib.astype(np.float16)); np.save(OUT / "embed" / "sim_nonsibling_sample.npy", non.astype(np.float16))
    (OUT / "embed" / "E2_embedding.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))

# %% 9. E3 — matched contamination study (frozen backbone, then fine-tuned)
def e3(tag, seeds, fine_tune, epochs):
    for seed in seeds:
        fold = stratified_group_folds(work.group, work.label, N_FOLDS, seed=100 + seed)
        for k in range(N_FOLDS):
            d = matched_design(work, fold, k, N_FOLDS, seed=1000 * seed + k)
            dname = OUT / "designs" / f"design_s{seed}_f{k}.csv"
            if not dname.exists():
                role = pd.Series("unused", index=work.index)
                role[d["base"]] = "base"; role[d["val"]] = "validation"; role[d["eval_original"]] = "eval_original"; role[d["eval_withheld"]] = "eval_withheld"
                role[np.setdiff1d(d["clean_train"], d["base"])] = "donor_bundle"; role[np.setdiff1d(d["exposed_train"], d["base"])] = "exposure_bundle"
                pd.DataFrame({"relative_path": work.relative_path, "role": role}).to_csv(dname, index=False)
                d["pairs"].to_csv(OUT / "designs" / f"pairs_s{seed}_f{k}.csv", index=False)
            ev = {"orig": d["eval_original"], "held": d["eval_withheld"], "val": d["val"]}
            pair = f"{tag}_s{seed}_f{k}"
            run_fit(f"{pair}_clean", d["clean_train"], d["val"], ev, fine_tune, seed * 100 + k, epochs, init_from=f"{pair}")
            run_fit(f"{pair}_exposed", d["exposed_train"], d["val"], ev, fine_tune, seed * 100 + k, epochs, init_from=f"{pair}")
if RUN_E3:
    e3("E3frozen", E3_SEEDS, False, E3_EPOCHS_FROZEN)
if RUN_E3_FT:
    e3("E3ft", E3_FT_SEEDS, True, E3_EPOCHS_FT)

# %% 10. E4 — repeated protocol comparison (image-level vs photo-level, early stopping as in v1)
if RUN_E4:
    from sklearn.model_selection import train_test_split
    n = len(work); idx = np.arange(n)
    for r in range(E4_REPEATS):
        tr, te_ = train_test_split(idx, test_size=0.15, stratify=Y, random_state=r)
        tr, va = train_test_split(tr, test_size=0.15 / 0.85, stratify=Y[tr], random_state=r)
        run_fit(f"E4image_r{r}", tr, va, {"test": te_}, False, 500 + r, E4_MAX_EPOCHS, early_stop=True)
        fold = stratified_group_folds(work.group, work.label, 20, seed=500 + r)
        te_g = idx[fold < 3]; va_g = idx[(fold >= 3) & (fold < 6)]; tr_g = idx[fold >= 6]
        run_fit(f"E4photo_r{r}", tr_g, va_g, {"test": te_g}, False, 500 + r, E4_MAX_EPOCHS, early_stop=True)

# %% 11. E5 — v1 seed-42 memberships re-run with the documented preprocessing
if RUN_E5:
    pos = {p: k for k, p in enumerate(work.relative_path)}
    for arm, url in V1_MANIFEST_URLS.items():
        m = pd.read_csv(url); rows = m.relative_path.map(pos).to_numpy()
        part = {s: rows[(m.split == s).to_numpy()] for s in ["train", "validation", "test"]}
        run_fit(f"E5v1_{arm}", part["train"], part["validation"], {"test": part["test"]}, False, 42, E4_MAX_EPOCHS, early_stop=True)

# %% 11b. Hashes of the saved initial-weight files (one file per E3 pair, loaded by both arms)
rows = [{"file": p.name, "sha256": sha256_file(p)} for p in sorted((OUT / "runs").glob("*.init_gpuaug.weights.h5"))]
pd.DataFrame(rows).to_csv(OUT / "runs" / "init_weight_hashes.csv", index=False); print(len(rows), "initial-weight files hashed")

# %% 12. Environment record and archive
env = {"finished_utc": utc(), "platform": platform.platform(), "python": sys.version.split()[0], "tensorflow": tf.__version__,
       "keras": getattr(tf.keras, "__version__", "bundled"), "numpy": np.__version__, "pandas": pd.__version__,
       "gpu": [tf.config.experimental.get_device_details(g) for g in tf.config.list_physical_devices("GPU")],
       "preprocess": PREPROCESS, "mixed_precision": MIXED_PRECISION, "config": {k: globals()[k] for k in [
           "N_FOLDS", "E3_SEEDS", "E3_FT_SEEDS", "E3_EPOCHS_FROZEN", "E3_EPOCHS_FT", "LR_FROZEN", "LR_FT", "E4_REPEATS", "E4_MAX_EPOCHS", "E4_PATIENCE", "BATCH"]}}
import sklearn; env["scikit_learn"] = sklearn.__version__
(OUT / "environment_v2.json").write_text(json.dumps(env, indent=2, default=str))
(OUT / "requirements_lock_v2.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
zpath = Path("/content/fruit_v2_results.zip") if IN_COLAB else Path("fruit_v2_results.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and not p.name.endswith((".h5", ".npy")) or p.name in {"sim_sibling.npy", "sim_nonsibling_sample.npy"}:
            z.write(p, p.relative_to(OUT))
print("archive:", zpath, round(zpath.stat().st_size / 1e6, 1), "MB")
if IN_COLAB:
    from google.colab import files; files.download(str(zpath))
