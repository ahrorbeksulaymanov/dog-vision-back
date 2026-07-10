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
# HOW TO FIND YOUR EXACT PATH:
#   1. After adding the dataset, expand it in the left sidebar (Input panel)
#   2. The path is shown at the top, e.g. /kaggle/input/stanford-dogs-dataset/
#   3. Since the zip contains a top-level `data/` folder, the full path to
#      your breed folders is usually:
#        /kaggle/input/<dataset-slug>/data/Images
#   4. Update DATA_DIR below to match what you see.

DATA_DIR = "/kaggle/input/stanford-dogs-dataset/data/Images"

if not os.path.isdir(DATA_DIR):
    # Fallback: try to find it by scanning /kaggle/input for the first dir
    # with 120 subfolders.
    print(f"DATA_DIR not found at: {DATA_DIR}")
    print("Scanning /kaggle/input for a folder with 120 breed subfolders...")
    found = None
    for root, dirs, _ in os.walk("/kaggle/input"):
        if len(dirs) >= 120 and "Images" in os.path.basename(root):
            found = root
            break
        if len(dirs) >= 120:
            found = root
            break
    if found:
        DATA_DIR = found
        print(f"  Auto-detected: {DATA_DIR}")
    else:
        raise RuntimeError(
            f"\nCould not find your dataset. Expected: {DATA_DIR}\n"
            f"Steps to fix:\n"
            f"  1. In Kaggle, left sidebar → Input → click + Add Input\n"
            f"  2. Find the dataset you uploaded (e.g. 'stanford-dogs-dataset')\n"
            f"  3. Click the + to add it to this notebook\n"
            f"  4. Expand it in the sidebar to see the actual folder path\n"
            f"  5. Update DATA_DIR at the top of this script to match"
        )

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
        "unfreeze_at": 200,
        "lr": 5e-6,
    },
    "convnext": {
        "fn": tf.keras.applications.ConvNeXtTiny,
        "unfreeze_at": 180,
        "lr": 5e-6,
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


def build_data_pipeline(data_dir, batch_size, img_size, is_training=True):
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

    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

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

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def unfreeze_backbone(base_model, unfreeze_from_layer):
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    total = len(base_model.layers)
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

def train_backbone(backbone_name, output_path, epochs=30, progressive=True):
    cfg = BACKBONES[backbone_name]
    print(f"\n{'='*60}")
    print(f"TRAINING: {backbone_name}")
    print(f"  Output: {output_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Progressive: {progressive}")
    print(f"{'='*60}")

    # Phase 1: 224x224
    train_ds, val_ds, class_names, num_classes = build_data_pipeline(
        DATA_DIR, BATCH_SIZE, 224, is_training=True
    )

    # Save class names (only once)
    labels_path = "/kaggle/working/unique_breeds.json"
    if not os.path.exists(labels_path):
        with open(labels_path, "w") as f:
            json.dump(class_names, f)
        print(f"  Saved {len(class_names)} breed labels to {labels_path}")

    model, base_model = build_model(num_classes, backbone_name, img_size=224)
    model.summary()

    # Step A: Train head
    print(f"\n--- STEP A: Training classifier head (frozen backbone) ---")
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
        tf.keras.callbacks.CSVLogger(f"/kaggle/working/training_log_{backbone_name}.csv"),
    ]
    head_epochs = max(5, epochs // 4)
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=callbacks, verbose=1)

    # Step B: Fine-tune
    print(f"\n--- STEP B: Fine-tuning (partial unfreeze) ---")
    unfreeze_backbone(base_model, cfg["unfreeze_at"])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    remaining_epochs = epochs - head_epochs
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=remaining_epochs,
        callbacks=callbacks + [
            tf.keras.callbacks.ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
        verbose=1,
    )

    # Phase 2: Progressive resizing to 384
    if progressive:
        print(f"\n--- PHASE 2: Progressive resizing to 384x384 ---")
        train_ds_384, val_ds_384, _, _ = build_data_pipeline(DATA_DIR, BATCH_SIZE // 2, 384, is_training=True)

        model_384, base_model_384 = build_model(num_classes, backbone_name, img_size=384)

        for layer_224 in model.layers:
            for layer_384 in model_384.layers:
                if layer_224.name == layer_384.name and layer_224.weights:
                    layer_384.set_weights(layer_224.get_weights())
                    break

        model = model_384
        base_model = base_model_384
        unfreeze_backbone(base_model, cfg["unfreeze_at"])
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

    # Save model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    print(f"  Model saved to {output_path}")

    # Evaluate
    val_ds_final = val_ds_384 if progressive else val_ds
    val_loss, val_acc, val_top3, val_top5 = model.evaluate(val_ds_final, verbose=0)
    print(f"\n  Validation Accuracy:  {val_acc:.2%}")
    print(f"  Validation Top-3 Acc: {val_top3:.2%}")
    print(f"  Validation Top-5 Acc: {val_top5:.2%}")
    print(f"  Validation Loss:      {val_loss:.4f}")

    # Temperature scaling
    print(f"\n--- TEMPERATURE SCALING ---")
    temperature = fit_temperature_scaling(model, val_ds_final)
    temp_path = output_path.replace(".h5", "_temp_scale.json")
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f)
    print(f"  Temperature: {temperature:.4f}")
    print(f"  Saved to {temp_path}")

    return val_acc, val_top3, val_top5


# ── 5. Run Training ─────────────────────────────────────────────────────

print("=" * 60)
print("STARTING KAGGLE GPU TRAINING")
print("=" * 60)

# Model 1: EfficientNetV2S (best single model)
acc1, top3_1, top5_1 = train_backbone(
    backbone_name="efficientnetv2s",
    output_path="/kaggle/working/dog_model_improved.h5",
    epochs=30,
    progressive=True,
)

# Model 2: ConvNeXtTiny (diversity for ensemble)
acc2, top3_2, top5_2 = train_backbone(
    backbone_name="convnext",
    output_path="/kaggle/working/dog_model_convnext.h5",
    epochs=25,
    progressive=True,
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