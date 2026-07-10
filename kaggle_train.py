"""
Dog Vision — Kaggle GPU Training Script
=======================================

How to use:
  1. Go to kaggle.com → Create → New Notebook
  2. Settings → Accelerator → GPU T4 x2  (or P100)
  3. Add Data → search "stanford dogs dataset" → add any version
  4. Copy this entire file into a single cell (or multiple cells)
  5. Run All
  6. After training, download the output files:
       - dog_model_improved.h5
       - dog_model_convnext.h5  (second backbone for ensemble)
       - temp_scale.json
       - unique_breeds.json
  7. Place them in your local models/ and data/ directories

This script trains TWO backbones for an ensemble:
  1. EfficientNetV2S + progressive resizing  (best single model)
  2. ConvNeXtTiny                             (diversity for ensemble)

Both include MixUp/CutMix, label smoothing, fine-tuning, and
temperature scaling for confidence calibration.

Environment: Kaggle GPU notebook (T4 or P100, ~30 hrs/week free)
Dataset: Stanford Dogs Dataset (120 breeds, ~20,580 images)
Expected time: ~45-90 min per backbone on T4 GPU
"""

import json
import os

import numpy as np
import tensorflow as tf

# ── 0. Setup & GPU Check ────────────────────────────────────────────────

GPUs = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(GPUs)}")
for gpu in GPUs:
    print(f"  {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)

if not GPUs:
    raise RuntimeError("No GPU detected. Enable GPU in Kaggle: Settings → Accelerator → GPU")

print(f"TensorFlow version: {tf.__version__}")


# ── 1. Locate & Prepare the Dataset ─────────────────────────────────────
#
# You uploaded your own zip of data/Images/ (120 snake_case breed folders).
# Kaggle mounts it under /kaggle/input/<your-dataset-name>/...
#
# Your actual path is:
#   /kaggle/input/datasets/mozzamshahid/stanford-dogs-dataset/data/Images
#
# If the path below doesn't exist, the script scans /kaggle/input to find
# the right one automatically. If it still can't find your data, it prints
# every directory it sees so you can pick the correct path.

DATA_DIR = "/kaggle/input/datasets/mozzamshahid/stanford-dogs-dataset/data/Images"

if not os.path.isdir(DATA_DIR):
    print(f"Primary path not found: {DATA_DIR}")
    print("Scanning /kaggle/input for a folder containing breed subfolders...")
    found = None

    for root, dirs, _ in os.walk("/kaggle/input"):
        if len(dirs) >= 100:
            found = root
            print(f"  Found {len(dirs)} subfolders in: {root}")
            break

    if found is None:
        print("\nCouldn't auto-detect. Here are all directories under /kaggle/input:")
        for root, dirs, _ in os.walk("/kaggle/input"):
            for d in dirs:
                print(f"  {os.path.join(root, d)}")
        raise RuntimeError(
            "\nCould not find your dataset. Steps to fix:\n"
            "  1. Left sidebar → Input → click + Add Input\n"
            "  2. Find 'mozzamshahid/stanford-dogs-dataset' (or whatever you named it)\n"
            "  3. Click the + to add it to this notebook\n"
            "  4. Expand it in the sidebar to see the actual folder path\n"
            "  5. Update DATA_DIR at the top of this script to match"
        )

    DATA_DIR = found
    print(f"\nUsing: {DATA_DIR}")

breed_folders = sorted(os.listdir(DATA_DIR))
print(f"\nDataset ready at: {DATA_DIR}")
print(f"  Breed folders: {len(breed_folders)}")
print(f"  First 3: {breed_folders[:3]}")
print(f"  Last 3:  {breed_folders[-3:]}")


# ── 2. (Renaming skipped — your folders are already snake_case) ──────────
# Your local zip used the clean labels (affenpinscher/, chihuahua/, etc.),
# so the script uses DATA_DIR directly. No symlink/rename pass needed.


# ── 3. Training Pipeline (adapted from train.py) ────────────────────────

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

BACKBONES = {
    "efficientnetv2s": {
        "fn": tf.keras.applications.EfficientNetV2S,
        "unfreeze_frac": 0.3,  # unfreeze the last 30% of backbone layers
        "lr": 1e-5,
    },
    "convnext": {
        "fn": tf.keras.applications.ConvNeXtTiny,
        "unfreeze_frac": 0.3,
        "lr": 1e-5,
    },
}


def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lambda_ = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    lambda_ = tf.maximum(lambda_, 1.0 - lambda_)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lambda_ * images + (1.0 - lambda_) * tf.gather(images, indices)
    mixed_labels = lambda_[:, :, 0, 0] * labels + (1.0 - lambda_[:, :, 0, 0]) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]
    lambda_ = tf.random.uniform([batch_size], 0.0, 1.0)
    lambda_ = tf.maximum(lambda_, 1.0 - lambda_)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(1.0 - lambda_), tf.int32)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(1.0 - lambda_), tf.int32)
    cx = tf.random.uniform([batch_size], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([batch_size], 0, img_h, dtype=tf.int32)
    half_w = tf.maximum(cut_w // 2, 1)
    half_h = tf.maximum(cut_h // 2, 1)
    x1 = tf.maximum(cx - half_w, 0)
    y1 = tf.maximum(cy - half_h, 0)
    x2 = tf.minimum(cx + half_w, img_w)
    y2 = tf.minimum(cy + half_h, img_h)
    indices = tf.random.shuffle(tf.range(batch_size))

    def build_mask(args):
        x1_i, x2_i, y1_i, y2_i = args
        y_grid = tf.range(img_h, dtype=tf.int32)
        x_grid = tf.range(img_w, dtype=tf.int32)
        yy, xx = tf.meshgrid(y_grid, x_grid, indexing="ij")
        inside = tf.logical_and(
            tf.logical_and(xx >= x1_i, xx < x2_i),
            tf.logical_and(yy >= y1_i, yy < y2_i),
        )
        return tf.cast(tf.logical_not(inside), tf.float32)

    mask = tf.map_fn(build_mask, (x1, x2, y1, y2), dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    shuffled_images = tf.gather(images, indices)
    mixed_images = mask * images + (1.0 - mask) * shuffled_images
    actual_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    total_area = tf.cast(img_w * img_h, tf.float32)
    actual_lambda = 1.0 - actual_area / total_area
    actual_lambda = tf.expand_dims(actual_lambda, axis=1)
    shuffled_labels = tf.gather(labels, indices)
    mixed_labels = actual_lambda * labels + (1.0 - actual_lambda) * shuffled_labels
    return mixed_images, mixed_labels


def build_data_pipeline(data_dir, batch_size, img_size, is_training=True, use_mixup=True):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="both",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=is_training,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Classes: {num_classes} breeds")
    print(f"  Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")
    print(f"  MixUp/CutMix: {use_mixup and is_training}")

    # IMPORTANT — pixel scaling order:
    # Augmentation runs on RAW [0, 255] pixels because RandomBrightness
    # defaults to value_range=(0, 255). Rescaling to [0, 1] BEFORE it (the
    # old bug) made brightness shifts of ±38 on [0, 1] images — pure white/
    # black garbage — which is why training produced terrible results.
    if is_training:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Scale to [0, 1] AFTER augmentation, matching the API's
    # preprocess_image(). The model itself rescales back to [0, 255]
    # internally — see build_model().
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # MixUp/CutMix only when use_mixup=True (skip in Step A so the head
        # can learn from clean labels first).
        if use_mixup:
            def apply_mixup(images, labels):
                if tf.random.uniform([]) < 0.5:
                    return mixup_batch(images, labels)
                else:
                    return cutmix_batch(images, labels)
            train_ds = train_ds.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names, num_classes


def build_model(num_classes, backbone_name, img_size=224):
    cfg = BACKBONES[backbone_name]
    base_model = cfg["fn"](
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # EfficientNetV2 and ConvNeXt EMBED their own normalization inside the
    # backbone and expect raw [0, 255] pixels (their preprocess_input is a
    # no-op placeholder). The old code fed them [0, 1] images — inputs 255x
    # smaller than expected — so the backbone produced garbage features and
    # accuracy collapsed. The model input stays [0, 1] (what the API sends);
    # Rescaling(255) is baked into the graph so the saved .h5 just works.
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def unfreeze_backbone(base_model, unfreeze_frac):
    # Unfreeze by FRACTION, not a hardcoded layer index: layer counts differ
    # per backbone (a fixed index like 180 could freeze the entire ConvNeXt,
    # leaving nothing to fine-tune).
    base_model.trainable = True
    total = len(base_model.layers)
    freeze_until = int(total * (1.0 - unfreeze_frac))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Backbone: {trainable}/{total} layers trainable")


def fit_temperature_scaling(model, val_ds):
    all_probs = []
    all_labels = []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)
        all_labels.append(labels_np)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    log_probs = np.log(all_probs + 1e-8)

    def compute_nll(T):
        scaled = log_probs / T
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        cal_probs = np.exp(scaled)
        cal_probs = cal_probs / np.sum(cal_probs, axis=1, keepdims=True)
        return -np.mean(np.sum(all_labels * np.log(cal_probs + 1e-8), axis=1))

    best_T = 1.0
    best_nll = compute_nll(1.0)
    nll_before = best_nll
    for T in np.arange(0.5, 5.01, 0.01):
        nll = compute_nll(T)
        if nll < best_nll:
            best_nll = nll
            best_T = float(round(T, 2))

    print(f"  NLL before (T=1.0): {nll_before:.4f}")
    print(f"  NLL after  (T={best_T:.2f}): {best_nll:.4f}")
    return best_T


# ── 4. Train a Single Backbone ───────────────────────────────────────────

def train_backbone(backbone_name, output_path, epochs=20, progressive=True):
    cfg = BACKBONES[backbone_name]
    print(f"\n{'='*60}")
    print(f"TRAINING: {backbone_name}")
    print(f"  Output: {output_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Progressive: {progressive}")
    print(f"{'='*60}")

    # Phase 1: 224x224, NO MixUp (so the head can learn from clean labels).
    train_ds, val_ds, class_names, num_classes = build_data_pipeline(
        DATA_DIR, BATCH_SIZE, 224, is_training=True, use_mixup=False,
    )

    # Save class names (only once)
    labels_path = "/kaggle/working/unique_breeds.json"
    if not os.path.exists(labels_path):
        with open(labels_path, "w") as f:
            json.dump(class_names, f)
        print(f"  Saved {len(class_names)} breed labels to {labels_path}")

    model, base_model = build_model(num_classes, backbone_name, img_size=224)
    model.summary()

    # Step A: Train head (NO MixUp, clean labels).
    print(f"\n--- STEP A: Training classifier head (frozen backbone, NO MixUp) ---")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(f"/kaggle/working/training_log_{backbone_name}.csv", append=True),
    ]
    head_epochs = max(5, epochs // 4)
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks, verbose=1)

    # Step B: Fine-tune. Re-enable MixUp for regularization once the head
    # has learned basic features. Rebuild the data pipeline to include MixUp.
    print(f"\n--- STEP B: Fine-tuning (partial unfreeze, MixUp re-enabled) ---")
    train_ds_mix, val_ds_mix, _, _ = build_data_pipeline(
        DATA_DIR, BATCH_SIZE, 224, is_training=True, use_mixup=True,
    )
    unfreeze_backbone(base_model, cfg["unfreeze_frac"])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    remaining_epochs = epochs - head_epochs
    model.fit(
        train_ds_mix, validation_data=val_ds_mix,
        epochs=remaining_epochs,
        callbacks=callbacks + [
            tf.keras.callbacks.ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
        verbose=1,
    )

    # ── Save 224 model NOW, before progressive 384, so the good
    # 224 weights are preserved even if 384 transfer goes wrong.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_224_path = output_path
    print(f"  224 model saved to {final_224_path} (preserved before 384 phase)")

    # Evaluate the 224 model
    val_loss, val_acc, val_top3, val_top5 = model.evaluate(val_ds_mix, verbose=0)
    print(f"\n  [224] Validation Accuracy:  {val_acc:.2%}")
    print(f"  [224] Validation Top-3 Acc: {val_top3:.2%}")
    print(f"  [224] Validation Top-5 Acc: {val_top5:.2%}")

    # Temperature scaling on 224 model
    print(f"\n--- TEMPERATURE SCALING (224) ---")
    temperature = fit_temperature_scaling(model, val_ds_mix)
    temp_path = output_path.replace(".h5", "_temp_scale.json")
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f)
    print(f"  Temperature: {temperature:.4f}")
    print(f"  Saved to {temp_path}")

    # Phase 2: Progressive resizing to 384
    if progressive:
        print(f"\n--- PHASE 2: Progressive resizing to 384x384 ---")
        train_ds_384, val_ds_384, _, _ = build_data_pipeline(DATA_DIR, BATCH_SIZE // 2, 384, is_training=True, use_mixup=True)

        model_384, base_model_384 = build_model(num_classes, backbone_name, img_size=384)

        # Copy weights by POSITION, not by name: Keras uniquifies layer names
        # across models built in the same process (dense → dense_2, ...), so
        # name-matching silently copies nothing and the 384 phase would train
        # from scratch. Identical architecture + fully-conv backbone means
        # the flat weight lists line up exactly regardless of input size.
        model_384.set_weights(model.get_weights())

        model = model_384
        base_model = base_model_384
        unfreeze_backbone(base_model, cfg["unfreeze_frac"])
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg["lr"] / 2, weight_decay=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy",
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                     tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
        )
        output_384 = output_path.replace(".h5", "_384.h5")
        model.fit(
            train_ds_384, validation_data=val_ds_384,
            epochs=epochs // 3,
            callbacks=callbacks + [
                tf.keras.callbacks.ModelCheckpoint(output_384, monitor="val_accuracy", save_best_only=True, verbose=1),
            ],
            verbose=1,
        )

        val_loss, val_acc, val_top3, val_top5 = model.evaluate(val_ds_384, verbose=0)
        print(f"\n  [384] Validation Accuracy:  {val_acc:.2%}")
        print(f"  [384] Validation Top-3 Acc: {val_top3:.2%}")
        print(f"  [384] Validation Top-5 Acc: {val_top5:.2%}")

    # Always re-evaluate the 224 model from disk so we report the saved
    # weights (not whatever's in memory after the 384 phase).
    from tensorflow.keras.models import load_model
    print(f"\n--- Re-evaluating saved 224 model from {final_224_path} ---")
    model_224_eval = load_model(final_224_path)
    val_loss, val_acc, val_top3, val_top5 = model_224_eval.evaluate(val_ds_mix, verbose=0)
    print(f"  [224 saved] Validation Accuracy:  {val_acc:.2%}")
    print(f"  [224 saved] Validation Top-3 Acc: {val_top3:.2%}")
    print(f"  [224 saved] Validation Top-5 Acc: {val_top5:.2%}")

    return val_acc, val_top3, val_top5


# ── 5. Run Training ─────────────────────────────────────────────────────

print("=" * 60)
print("STARTING KAGGLE GPU TRAINING")
print("=" * 60)

# The API serves 224x224, so the 384 phase is optional polish (~+1%) that
# roughly DOUBLES training time. Off by default — flip it on if you have
# GPU hours to spare. ConvNeXt gives ensemble diversity; also optional.
PROGRESSIVE_384 = False
TRAIN_CONVNEXT = True

# Model 1: EfficientNetV2S (best single model)
acc1, top3_1, top5_1 = train_backbone(
    backbone_name="efficientnetv2s",
    output_path="/kaggle/working/dog_model_improved.h5",
    epochs=15,
    progressive=PROGRESSIVE_384,
)

# Model 2: ConvNeXtTiny (diversity for ensemble)
if TRAIN_CONVNEXT:
    acc2, top3_2, top5_2 = train_backbone(
        backbone_name="convnext",
        output_path="/kaggle/working/dog_model_convnext.h5",
        epochs=12,
        progressive=PROGRESSIVE_384,
    )

# ── 6. Summary ──────────────────────────────────────────────────────────

# ── Save unified temp_scale.json for the predictor ──────────────────────

with open("/kaggle/working/dog_model_improved_temp_scale.json") as f:
    eff_temp = json.load(f)["temperature"]

with open("/kaggle/working/temp_scale.json", "w") as f:
    json.dump({"temperature": eff_temp}, f)
print(f"\nSaved unified temp_scale.json (T={eff_temp:.4f})")

print("\n" + "=" * 60)
print("TRAINING COMPLETE — RESULTS SUMMARY")
print("=" * 60)
print(f"  Model 1 (EfficientNetV2S):  acc={acc1:.2%}  top3={top3_1:.2%}  top5={top5_1:.2%}")
if TRAIN_CONVNEXT:
    print(f"  Model 2 (ConvNeXtTiny):     acc={acc2:.2%}  top3={top3_2:.2%}  top5={top5_2:.2%}")
print(f"\nOutput files in /kaggle/working/:")
for f in sorted(os.listdir("/kaggle/working/")):
    if f.endswith(".h5") or f.endswith(".json"):
        size = os.path.getsize(os.path.join("/kaggle/working/", f))
        print(f"  {f}  ({size / 1e6:.1f} MB)")
print("\nDownload these files and place them in your local repo:")
print("  dog_model_improved.h5        → models/")
print("  dog_model_convnext.h5        → models/")
print("  temp_scale.json              → models/")
print("  unique_breeds.json           → data/")
print("=" * 60)