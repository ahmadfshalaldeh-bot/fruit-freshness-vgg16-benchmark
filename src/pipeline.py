# %% [markdown]
# # VGG16 fresh/rotten fruit benchmark — full reproducibility pipeline
#
# Reruns the experiment described in the Applied Sciences manuscript
# *"VGG16-Based Classification of Fresh and Rotten Apple, Banana, and Orange Images in a Public Dataset"*
# and writes every supplementary item the manuscript promises (S1–S3, Code S1–S2, Figure S1, Figure 3, Table 3).
#
# **How to run (Google Colab):** Runtime → Change runtime type → GPU (T4 or better) → Runtime → Run all.
# **How to run (local):** `pip install tensorflow scikit-learn pandas matplotlib pillow scipy kagglehub` then `python fruit_vgg16_reproducibility.py`.
#
# Two evaluation protocols are supported (set SPLIT_MODE in cell 0):
# * "image"  – conventional stratified image-level split (what most published work on this dataset does);
# * "family" – photo-level split: the 9 stored versions of each original photograph (original, 5 rotations,
#              salt-and-pepper, translation, vertical flip) are kept together in one partition.
# Report whatever the run produces.

# %% 0. Configuration — edit only this cell
from pathlib import Path
import os, sys

SEED = 42
IMG_SIZE = 224
PER_CLASS = None          # None = every unique image (13,599). An integer subsamples each class.
DROP_EXACT_DUPLICATES = True   # remove byte-identical copies (SHA-256) before sampling
SPLIT_MODE = "family"     # "family" = photo-level split (recommended); "image" = image-level split; "grouped" = pHash clusters
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-4
PHASH_MAX_DISTANCE = 6    # Hamming distance (of 64 bits) counted as a near-duplicate; rotations/flips included
INCLUDE_WEIGHTS_IN_ARCHIVE = True
DETERMINISTIC_OPS = True  # slower but repeatable on the same hardware/software

KAGGLE_DATASET = "sriramr/fruits-fresh-and-rotten-for-classification"
DATA_DIR = None           # set to an already-extracted folder to skip the download

# Class order used everywhere (rows/columns of the confusion matrix)
CLASSES = ["freshapples", "rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges"]
CLASS_NAMES = ["Fresh apple", "Rotten apple", "Fresh banana", "Rotten banana", "Fresh orange", "Rotten orange"]

IN_COLAB = "google.colab" in sys.modules
OUT = (Path("/content") if IN_COLAB else Path(".")) / f"repro_output_{SPLIT_MODE}"
OUT.mkdir(parents=True, exist_ok=True)
print("Output folder:", OUT.resolve())

# %% 1. Seeds and imports
import json, hashlib, platform, subprocess, datetime, random, zipfile, math
import numpy as np
import pandas as pd
from PIL import Image

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

# %% 2. Download the dataset and record provenance
def get_dataset_root():
    if DATA_DIR:
        return Path(DATA_DIR), {"source": "local folder", "path": str(DATA_DIR)}
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return Path(path), {
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "kagglehub_path": str(path),   # contains the dataset version number, e.g. .../versions/1
        "kagglehub_version": getattr(kagglehub, "__version__", "unknown"),
    }

DATA_ROOT, provenance = get_dataset_root()
provenance["downloaded_utc"] = utc_now()
print(json.dumps(provenance, indent=2))

# %% 3. Inventory every image file (manifest, raw)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
class_lookup = {c.lower(): c for c in CLASSES}

rows = []
for p in sorted(DATA_ROOT.rglob("*")):
    if p.suffix.lower() in IMG_EXT and p.parent.name.lower() in class_lookup:
        rel = p.relative_to(DATA_ROOT).as_posix()
        parts = [s.lower() for s in p.relative_to(DATA_ROOT).parts]
        source_split = "train" if "train" in parts else ("test" if "test" in parts else "unspecified")
        rows.append({"path": str(p), "relative_path": rel,
                     "label": class_lookup[p.parent.name.lower()], "source_split": source_split})
inventory = pd.DataFrame(rows)
if inventory.empty:
    raise SystemExit(f"No images found under {DATA_ROOT}. Check DATA_DIR / the download.")

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

# --- perceptual hash (DCT pHash, 64 bit) for the 8 rotations/flips of each image ---
from scipy.fft import dctn

def _phash_bits(gray32):
    d = dctn(gray32, norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return np.packbits(bits).view(">u8")[0]

def phash_variants(img):
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    out = []
    for flip in (False, True):
        a = np.fliplr(g) if flip else g
        for k in range(4):
            out.append(_phash_bits(np.rot90(a, k)))
    return np.array(out, dtype=np.uint64)   # index 0 = original orientation

print(f"Hashing {len(inventory):,} files (SHA-256 + pHash) ...")
sha, w, h, mode, ph, readable = [], [], [], [], [], []
for pth in inventory["path"]:
    sha.append(sha256_file(pth))
    try:
        with Image.open(pth) as im:
            im.load()
            w.append(im.width); h.append(im.height); mode.append(im.mode)
            ph.append(phash_variants(im.convert("RGB")))
            readable.append(True)
    except Exception:
        w.append(None); h.append(None); mode.append(None); ph.append(np.zeros(8, np.uint64)); readable.append(False)
inventory["sha256"], inventory["width"], inventory["height"], inventory["mode"], inventory["readable"] = sha, w, h, mode, readable
PHASH_ALL = np.stack(ph)
inventory["phash_hex"] = [f"{int(v[0]):016x}" for v in PHASH_ALL]

# Provenance of each file: the public dataset stores each original photograph in 9 versions whose
# file names carry the operation as a prefix (e.g. "rotated_by_15_Screen Shot ... .png").
OP_RE = r"^(rotated_by_\d+|saltandpepper|translation|vertical_flip)_"
fname = inventory["relative_path"].str.split("/").str[-1]
inventory["variant"] = fname.str.extract(OP_RE)[0].fillna("original")
inventory["family"] = inventory["label"] + "|" + fname.str.replace(OP_RE, "", regex=True)

counts_raw = inventory.groupby(["label", "source_split"]).size().unstack(fill_value=0)
print("\nRaw file counts (as downloaded):\n", counts_raw)
inventory.to_csv(OUT / "S1a_raw_inventory.csv", index=False)

# %% 4. Build the analysis set (exact-duplicate removal + per-class sampling)
work = inventory[inventory["readable"]].copy()
work["phash_idx"] = work.index
n_exact_dupes = int(work.duplicated("sha256").sum())
cross_label_exact = work.groupby("sha256")["label"].nunique()
cross_label_exact = int((cross_label_exact > 1).sum())
if DROP_EXACT_DUPLICATES:
    work = work.sort_values("relative_path").drop_duplicates("sha256", keep="first")

unique_counts = work["label"].value_counts().reindex(CLASSES).fillna(0).astype(int)
print("\nUnique readable images per class:\n", unique_counts.to_string())
print(f"Exact (byte-identical) duplicates found: {n_exact_dupes:,}; hashes shared across different labels: {cross_label_exact}")

if PER_CLASS is not None:
    short = unique_counts[unique_counts < PER_CLASS]
    if len(short):
        raise SystemExit(
            "\n*** STOP: these classes have fewer than PER_CLASS unique images: "
            f"{short.to_dict()}.\nThe manuscript's '2,000 images per class / 12,000 total' cannot be built from "
            "this download. Decide with your co-authors: set PER_CLASS = None (use all images, update the "
            "manuscript counts) or set PER_CLASS to the smallest class size, then rerun.")
    work = work.groupby("label").sample(n=PER_CLASS, random_state=SEED).reset_index(drop=True)
work["label_idx"] = work["label"].map({c: i for i, c in enumerate(CLASSES)})
print(f"\nAnalysis set: {len(work):,} images")

# %% 5. Near-duplicate audit (perceptual hash, rotation/flip-invariant)
def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)

H = PHASH_ALL[work["phash_idx"].to_numpy()]      # (n, 8)
base = H[:, 0]                                    # (n,)
n = len(work)
pairs = []
CH = 128
for s in range(0, n, CH):
    blk = H[s:s + CH]                                             # (b, 8)
    d = popcount64(blk[:, None, :] ^ base[None, :, None]).min(-1)  # (b, n) min over 8 variants
    ii, jj = np.nonzero(d <= PHASH_MAX_DISTANCE)
    ii = ii + s
    keep = jj > ii
    for i, j in zip(ii[keep], jj[keep]):
        pairs.append((int(i), int(j), int(d[i - s, j])))

parent = list(range(n))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i, j, _ in pairs:
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[rj] = ri
work["near_dup_group"] = [find(i) for i in range(n)]
group_sizes = work["near_dup_group"].map(work["near_dup_group"].value_counts())
print(f"Near-duplicate pairs (Hamming <= {PHASH_MAX_DISTANCE}, any rotation/flip): {len(pairs):,}")
print(f"Images belonging to a near-duplicate cluster: {(group_sizes > 1).sum():,} of {n:,}")

# %% 6. Split (70 / 15 / 15, seed 42)
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

idx = np.arange(n)
y = work["label_idx"].to_numpy()
if SPLIT_MODE == "image":
    n_test = round(TEST_FRAC * n); n_val = round(VAL_FRAC * n)
    trval_idx, test_idx = train_test_split(idx, test_size=n_test, stratify=y, random_state=SEED)
    train_idx, val_idx = train_test_split(trval_idx, test_size=n_val, stratify=y[trval_idx], random_state=SEED)
elif SPLIT_MODE == "family":
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    folds = [te for _, te in sgkf.split(idx, y, groups=work["family"])]
    test_idx = np.concatenate(folds[:3]); val_idx = np.concatenate(folds[3:6]); train_idx = np.concatenate(folds[6:])
elif SPLIT_MODE == "grouped":
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    folds = [te for _, te in sgkf.split(idx, y, groups=work["near_dup_group"])]
    test_idx = np.concatenate(folds[:3]); val_idx = np.concatenate(folds[3:6]); train_idx = np.concatenate(folds[6:])
else:
    raise ValueError("SPLIT_MODE must be 'image', 'family' or 'grouped'")

split = np.empty(n, dtype=object)
split[train_idx], split[val_idx], split[test_idx] = "train", "validation", "test"
work["split"] = split
print(pd.crosstab(work["label"], work["split"]).reindex(CLASSES))

# Leakage: near-duplicate pairs that straddle partitions
sp = work["split"].to_numpy(); lab = work["label"].to_numpy()
pair_df = pd.DataFrame([{
    "file_a": work.at[i, "relative_path"], "file_b": work.at[j, "relative_path"],
    "label_a": lab[i], "label_b": lab[j], "split_a": sp[i], "split_b": sp[j],
    "hamming": dist, "cross_label": lab[i] != lab[j], "cross_split": sp[i] != sp[j],
} for i, j, dist in pairs])
pair_df.to_csv(OUT / "S2_near_duplicate_pairs.csv", index=False)
def straddle(a, b):
    if pair_df.empty: return 0
    return int((((pair_df.split_a == a) & (pair_df.split_b == b)) | ((pair_df.split_a == b) & (pair_df.split_b == a))).sum())
test_with_train_twin = 0
if not pair_df.empty:
    tw = pair_df[((pair_df.split_a == "test") & (pair_df.split_b == "train")) | ((pair_df.split_b == "test") & (pair_df.split_a == "train"))]
    test_with_train_twin = len(set(tw.loc[tw.split_a == "test", "file_a"]) | set(tw.loc[tw.split_b == "test", "file_b"]))
audit = {
    "phash_max_hamming_distance": PHASH_MAX_DISTANCE,
    "raw_files": int(len(inventory)), "unreadable_files": int((~inventory["readable"]).sum()),
    "exact_duplicate_files_removed": n_exact_dupes if DROP_EXACT_DUPLICATES else 0,
    "exact_hashes_with_conflicting_labels": cross_label_exact,
    "analysis_images": int(n),
    "near_duplicate_pairs": len(pairs),
    "near_duplicate_pairs_with_conflicting_labels": int(pair_df["cross_label"].sum()) if len(pair_df) else 0,
    "pairs_train_test": straddle("train", "test"), "pairs_train_validation": straddle("train", "validation"),
    "pairs_validation_test": straddle("validation", "test"),
    "test_images_with_near_duplicate_in_train": int(test_with_train_twin),
    "split_mode": SPLIT_MODE,
    "families": int(work["family"].nunique()),
    "test_images_with_same_photo_version_in_train": int(work[work.split == "test"]["family"].isin(set(work[work.split == "train"]["family"])).sum()),
    "families_split_across_partitions": int((work.groupby("family")["split"].nunique() > 1).sum()),
}
(OUT / "S2_duplicate_audit_summary.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))

manifest_cols = ["relative_path", "label", "variant", "family", "source_split", "sha256", "phash_hex", "width", "height", "mode", "near_dup_group", "split"]
work[manifest_cols].to_csv(OUT / "S1_manifest_with_split.csv", index=False)

# %% 7. TensorFlow input pipeline (augmentation exactly as declared in Section 2.2)
import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)
if DETERMINISTIC_OPS:
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Op determinism not available:", e)
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

def load_uint8(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8), label

def to_unit(img, label):  # scale 0–255 → [0, 1] as declared in the manuscript
    return tf.cast(img, tf.float32) / 255.0, tf.one_hot(label, len(CLASSES))

DEG = math.pi / 180.0
def augment(img, label):
    # Mirrors Keras ImageDataGenerator semantics: rotation ±20°, width/height shift 0.1,
    # shear_range 0.1 (degrees, as in ImageDataGenerator), zoom 0.8–1.2 per axis,
    # horizontal + vertical flips, brightness ×0.8–1.2, fill_mode='nearest'.
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    S = float(IMG_SIZE)
    th = tf.random.uniform([], -20.0, 20.0) * DEG
    sh = tf.random.uniform([], -0.1, 0.1) * DEG
    zx = tf.random.uniform([], 0.8, 1.2); zy = tf.random.uniform([], 0.8, 1.2)
    tx = tf.random.uniform([], -0.1, 0.1) * S; ty = tf.random.uniform([], -0.1, 0.1) * S
    c, s = tf.cos(th), tf.sin(th)
    one, zero = tf.constant(1.0), tf.constant(0.0)
    m3 = lambda *v: tf.reshape(tf.stack(v), [3, 3])
    R = m3(c, -s, zero, s, c, zero, zero, zero, one)
    SH = m3(one, -tf.sin(sh), zero, zero, tf.cos(sh), zero, zero, zero, one)
    Z = m3(zx, zero, zero, zero, zy, zero, zero, zero, one)
    T = m3(one, zero, tx, zero, one, ty, zero, zero, one)
    ctr = (S - 1.0) / 2.0
    C = tf.constant([[1.0, 0.0, ctr], [0.0, 1.0, ctr], [0.0, 0.0, 1.0]])
    Ci = tf.constant([[1.0, 0.0, -ctr], [0.0, 1.0, -ctr], [0.0, 0.0, 1.0]])
    M = C @ T @ R @ SH @ Z @ Ci       # maps output pixel → input pixel
    M = M / M[2, 2]
    tvec = tf.reshape(tf.stack([M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2], M[2, 0], M[2, 1]]), [1, 8])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=tvec, output_shape=tf.constant([IMG_SIZE, IMG_SIZE]),
        fill_value=0.0, interpolation="BILINEAR", fill_mode="NEAREST")[0]
    img = img * tf.random.uniform([], 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 1.0), label

AUTOTUNE = tf.data.AUTOTUNE
def make_ds(frame, training):
    ds = tf.data.Dataset.from_tensor_slices((frame["path"].tolist(), frame["label_idx"].to_numpy(np.int32)))
    ds = ds.map(load_uint8, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(to_unit, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_df = work[work.split == "train"].reset_index(drop=True)
val_df = work[work.split == "validation"].reset_index(drop=True)
test_df = work[work.split == "test"].reset_index(drop=True)
train_ds, val_ds, test_ds = make_ds(train_df, True), make_ds(val_df, False), make_ds(test_df, False)

# Save one augmented batch preview for your own sanity check (contains dataset photos — do NOT put in the paper)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xb, _ = next(iter(train_ds))
fig, axs = plt.subplots(2, 6, figsize=(12, 4.4))
for a, im in zip(axs.flat, xb[:12].numpy()):
    a.imshow(im); a.axis("off")
fig.savefig(OUT / "_private_augmentation_check.png", dpi=100, bbox_inches="tight"); plt.close(fig)

# %% 8. Model (Section 2.3 / Table 2)
base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False
inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
model = tf.keras.Model(inp, out, name="vgg16_fruit_six_class")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy", metrics=["accuracy"])

summary_lines = []
model.summary(print_fn=lambda s, **k: summary_lines.append(s))
(OUT / "S_Figure_S1_model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
print("\n".join(summary_lines))

# %% 9. Train
WEIGHTS = OUT / "vgg16_best.weights.h5"
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(str(WEIGHTS), monitor="val_loss", save_best_only=True, save_weights_only=True),
    tf.keras.callbacks.CSVLogger(str(OUT / "training_history.csv")),
]
t0 = datetime.datetime.now()
history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, callbacks=callbacks, verbose=2)
train_minutes = (datetime.datetime.now() - t0).total_seconds() / 60
hist = pd.DataFrame(history.history)
best_epoch = int(hist["val_loss"].idxmin()) + 1
print(f"Stopped after {len(hist)} epochs; best validation loss at epoch {best_epoch}; {train_minutes:.1f} min")

# Real training curves (supplementary — generated from the logged history, not drawn)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(hist.index + 1, hist["loss"], label="Training"); ax[0].plot(hist.index + 1, hist["val_loss"], label="Validation")
ax[0].axvline(best_epoch, ls="--", c="grey", lw=0.8); ax[0].set(xlabel="Epoch", ylabel="Categorical cross-entropy", title="(a) Loss")
ax[1].plot(hist.index + 1, hist["accuracy"], label="Training"); ax[1].plot(hist.index + 1, hist["val_accuracy"], label="Validation")
ax[1].axvline(best_epoch, ls="--", c="grey", lw=0.8); ax[1].set(xlabel="Epoch", ylabel="Accuracy", title="(b) Accuracy")
for a in ax: a.legend(frameon=False); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "Figure_S2_training_curves.png", dpi=600); plt.close(fig)

# %% 10. Held-out evaluation → Table 3, Figure 3, predictions (S3)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

probs = model.predict(test_ds, verbose=0)
y_true = test_df["label_idx"].to_numpy()
y_pred = probs.argmax(1)
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
acc = accuracy_score(y_true, y_pred)
mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

table3 = pd.DataFrame({"Class": CLASS_NAMES, "Test support": sup,
                       "Precision (%)": (p * 100).round(2), "Recall (%)": (r * 100).round(2), "F1-score (%)": (f * 100).round(2)})
table3.loc[len(table3)] = ["Macro average", int(sup.sum()), round(mp * 100, 2), round(mr * 100, 2), round(mf * 100, 2)]
table3.to_csv(OUT / "Table_3_class_metrics.csv", index=False)
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT / "confusion_matrix_counts.csv")

pred = test_df[["relative_path", "sha256", "label", "variant", "family"]].copy()
pred["predicted"] = [CLASSES[i] for i in y_pred]
pred["correct"] = pred["label"] == pred["predicted"]
for i, cname in enumerate(CLASSES):
    pred[f"prob_{cname}"] = probs[:, i].round(6)
pred.to_csv(OUT / "S3_heldout_predictions.csv", index=False)

correct = int((y_true == y_pred).sum())
print(table3.to_string(index=False))
print(f"\nHeld-out ({SPLIT_MODE} split): {correct}/{len(y_true)} correct = {acc * 100:.2f}% | macro P/R/F1 = {mp*100:.2f}/{mr*100:.2f}/{mf*100:.2f}%")

# Extra views: originals only (one image per photograph), per-variant accuracy, photo-clustered bootstrap CI
orig = (pred["variant"] == "original").to_numpy()
po, ro, fo, _ = precision_recall_fscore_support(y_true[orig], y_pred[orig], labels=range(len(CLASSES)), zero_division=0)
mpo, mro, mfo, _ = precision_recall_fscore_support(y_true[orig], y_pred[orig], average="macro", zero_division=0)
acc_orig = float((y_true[orig] == y_pred[orig]).mean())
per_variant = pred.groupby("variant")["correct"].agg(["sum", "count", "mean"])
per_variant.to_csv(OUT / "accuracy_by_variant.csv")
fam_codes, fam_index = np.unique(pred["family"], return_inverse=True)
rng = np.random.default_rng(SEED); boots_acc, boots_f1 = [], []
members = [np.where(fam_index == k)[0] for k in range(len(fam_codes))]
for _ in range(5000):
    pick = rng.integers(0, len(members), len(members)); ii = np.concatenate([members[k] for k in pick])
    boots_acc.append((y_true[ii] == y_pred[ii]).mean())
    boots_f1.append(precision_recall_fscore_support(y_true[ii], y_pred[ii], average="macro", zero_division=0)[2])
ci = {"accuracy_ci95_photo_bootstrap": [float(np.percentile(boots_acc, 2.5)), float(np.percentile(boots_acc, 97.5))],
      "macro_f1_ci95_photo_bootstrap": [float(np.percentile(boots_f1, 2.5)), float(np.percentile(boots_f1, 97.5))]}
pd.DataFrame({"Class": CLASS_NAMES, "Test support": np.bincount(y_true[orig], minlength=6), "Precision (%)": (po*100).round(2),
              "Recall (%)": (ro*100).round(2), "F1-score (%)": (fo*100).round(2)}).to_csv(OUT / "class_metrics_originals_only.csv", index=False)
print(f"Originals only: n={orig.sum()}, accuracy={acc_orig*100:.2f}%, macro F1={mfo*100:.2f}%")
print("Accuracy by stored version:\n", per_variant.to_string())
print("Photo-clustered bootstrap 95% CI:", ci)
verdict = {"originals_only": {"n": int(orig.sum()), "accuracy": acc_orig, "macro_precision": float(mpo), "macro_recall": float(mro), "macro_f1": float(mfo)}, **ci}

# Figure 3 — (a) confusion matrix, (b) class-wise metrics; 600 dpi, no dataset photographs
short = ["F-apple", "R-apple", "F-banana", "R-banana", "F-orange", "R-orange"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
im = ax[0].imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[0].set_xticks(range(6), short, rotation=45, ha="right"); ax[0].set_yticks(range(6), short)
ax[0].set(xlabel="Predicted label", ylabel="True label", title="(a) Confusion matrix (held-out test)")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
xw = np.arange(6); bw = 0.26
for k, (vals, name) in enumerate([(p, "Precision"), (r, "Recall"), (f, "F1-score")]):
    ax[1].bar(xw + (k - 1) * bw, vals * 100, bw, label=name)
lo = max(0.0, math.floor(min(p.min(), r.min(), f.min()) * 100) - 2)
ax[1].set_ylim(lo, 100.5); ax[1].set_xticks(xw, short, rotation=45, ha="right")
ax[1].set(ylabel="Score (%)", title="(b) Class-wise metrics"); ax[1].legend(frameon=False, ncol=3, loc="lower center")
ax[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "Figure_3_heldout_performance.png", dpi=600); plt.close(fig)

# %% 11. Environment record (Code S2) — use THESE values in Table 2
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"
gpus = tf.config.list_physical_devices("GPU")
gpu_details = [tf.config.experimental.get_device_details(g) for g in gpus]
import sklearn, scipy, matplotlib as mpl
env = {
    "run_finished_utc": utc_now(), "platform": platform.platform(), "python": sys.version,
    "tensorflow": tf.__version__, "keras": getattr(tf.keras, "__version__", "bundled"),
    "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__, "matplotlib": mpl.__version__, "pillow": Image.__version__,
    "gpus": gpu_details, "cpu": platform.processor() or run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo"]),
    "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    "cuda_build": tf.sysconfig.get_build_info().get("cuda_version"), "cudnn_build": tf.sysconfig.get_build_info().get("cudnn_version"),
    "colab": IN_COLAB,
}
(OUT / "S_Code_S2_environment.json").write_text(json.dumps(env, indent=2, default=str))
(OUT / "S_Code_S2_requirements_lock.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]))

run_record = {
    "provenance": provenance, "config": {
        "seed": SEED, "img_size": IMG_SIZE, "per_class": PER_CLASS, "drop_exact_duplicates": DROP_EXACT_DUPLICATES,
        "split_mode": SPLIT_MODE, "fractions": [TRAIN_FRAC, VAL_FRAC, TEST_FRAC], "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "learning_rate": LEARNING_RATE,
        "early_stopping_restore_best_weights": True, "pixel_scaling": "x/255 -> [0,1]",
        "augmentation": "rot ±20°, h+v flip, zoom 0.8–1.2, shift 0.1, shear 0.1° (ImageDataGenerator semantics), brightness ×0.8–1.2, fill nearest"},
    "split_counts": {k: int(v) for k, v in work["split"].value_counts().items()},
    "epochs_run": int(len(hist)), "best_epoch": best_epoch, "training_minutes": round(train_minutes, 1),
    "heldout": {"correct": correct, "total": int(len(y_true)), "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf},
    "extra_views": verdict, "duplicate_audit": audit,
}
(OUT / "run_record.json").write_text(json.dumps(run_record, indent=2, default=str))

# %% 12. Package the reproducibility archive (checksums + README)
script_copy = OUT / "Code_S1_pipeline.py"
try:
    src = Path(__file__)
    script_copy.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
except NameError:
    script_copy.write_text("# In Colab: File → Download → .ipynb and add the notebook to this archive.\n")

readme = f"""# Reproducibility archive — VGG16 six-class fresh/rotten fruit benchmark

Generated {utc_now()}. Dataset: {provenance.get('source')} (downloaded {provenance['downloaded_utc']}).
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

Held-out result: {correct}/{len(y_true)} = {acc*100:.2f}% accuracy; macro F1 = {mf*100:.2f}%.
To reproduce: run Code_S1 with the same configuration (seed {SEED}).
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

skip = {"_private_augmentation_check.png"} | (set() if INCLUDE_WEIGHTS_IN_ARCHIVE else {WEIGHTS.name})
files = sorted(q for q in OUT.iterdir() if q.is_file() and q.name not in skip and q.suffix != ".zip" and q.name != "SHA256SUMS.txt")
(OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(q)}  {q.name}\n" for q in files))
zpath = OUT / f"reproducibility_archive_{SPLIT_MODE}.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for q in files + [OUT / "SHA256SUMS.txt"]:
        z.write(q, q.name)
print("Archive written:", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")

if IN_COLAB:
    from google.colab import files as colab_files
    colab_files.download(str(zpath))
    print("Archive download started:", zpath.name)
