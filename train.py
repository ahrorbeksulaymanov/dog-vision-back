"""
Dog Breed Classifier — Training & Fine-tuning Script

Supports multiple backbones, MixUp/CutMix, progressive resizing, and more.

Usage:
    # Basic training with default MobileNetV2
    python train.py --data_dir /path/to/dataset

    # With EfficientNetV2 (better accuracy)
    python train.py --data_dir /path/to/dataset --backbone efficientnetv2

    # With progressive resizing (best accuracy)
    python train.py --data_dir /path/to/dataset --backbone efficientnetv2 --progressive

Dataset structure:
    data_dir/
    ├── affenpinscher/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── afghan_hound/
    │   └── ...
    └── ...

The Stanford Dogs Dataset can be downloaded from:
    http://vision.stanford.edu/aditya86/ImageNetDogs/
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
DEFAULT_DATA_DIR = "data/Images"

BACKBONES = {
    "mobilenetv2": {
        "fn": tf.keras.applications.MobileNetV2,
        "size": 224,
        "unfreeze_at": 100,
        "lr": 1e-5,
    },
    "efficientnetv2": {
        "fn": tf.keras.applications.EfficientNetV2B0,
        "size": 224,
        "unfreeze_at": 150,
        "lr": 5e-6,
    },
    "efficientnetv2s": {
        "fn": tf.keras.applications.EfficientNetV2S,
        "size": 224,
        "unfreeze_at": 200,
        "lr": 5e-6,
    },
    "convnext": {
        "fn": tf.keras.applications.ConvNeXtTiny,
        "size": 224,
        "unfreeze_at": 180,
        "lr": 5e-6,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dog Breed Classifier")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to dataset directory (organized by breed subfolders)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size (default: 32)")
    parser.add_argument("--backbone", default="mobilenetv2",
                        choices=list(BACKBONES.keys()),
                        help="Backbone architecture (default: mobilenetv2)")
    parser.add_argument("--progressive", action="store_true",
                        help="Use progressive resizing: 224 → 384")
    parser.add_argument("--mixup", action="store_true", default=True,
                        help="Use MixUp augmentation (default: True)")
    parser.add_argument("--no-mixup", action="store_true",
                        help="Disable MixUp augmentation")
    parser.add_argument("--output", default="models/dog_model_improved.h5",
                        help="Path to save the trained model")
    parser.add_argument("--resume", default=None,
                        help="Path to existing model to resume training from")
    return parser.parse_args()


# ── MixUp & CutMix ─────────────────────────────────────────────────────

def mixup_batch(images, labels, alpha=0.2):
    """
    MixUp: blend two images and their labels linearly.

    λ ~ Beta(α, α). New image = λ * img1 + (1-λ) * img2.
    """
    batch_size = tf.shape(images)[0]
    lambda_ = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    lambda_ = tf.maximum(lambda_, 1.0 - lambda_)  # Symmetric

    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lambda_ * images + (1.0 - lambda_) * tf.gather(images, indices)
    mixed_labels = lambda_[:, :, 0, 0] * labels + (1.0 - lambda_[:, :, 0, 0]) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=0.2):
    """
    CutMix: cut a patch from one image and paste it onto another.

    Uses a vectorized mask approach with tf.map_fn.
    """
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    # Sample λ ~ Uniform(0, 1)
    lambda_ = tf.random.uniform([batch_size], 0.0, 1.0)
    lambda_ = tf.maximum(lambda_, 1.0 - lambda_)

    # Box dimensions
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(1.0 - lambda_), tf.int32)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(1.0 - lambda_), tf.int32)

    # Random center points
    cx = tf.random.uniform([batch_size], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([batch_size], 0, img_h, dtype=tf.int32)

    # Compute box boundaries (clamped)
    half_w = tf.maximum(cut_w // 2, 1)
    half_h = tf.maximum(cut_h // 2, 1)
    x1 = tf.maximum(cx - half_w, 0)
    y1 = tf.maximum(cy - half_h, 0)
    x2 = tf.minimum(cx + half_w, img_w)
    y2 = tf.minimum(cy + half_h, img_h)

    # Shuffle for source images
    indices = tf.random.shuffle(tf.range(batch_size))

    # Build mask using tf.map_fn (vectorized over batch)
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

    mask = tf.map_fn(build_mask, (x1, x2, y1, y2), dtype=tf.float32)  # [B, H, W]
    mask = tf.expand_dims(mask, axis=-1)  # [B, H, W, 1]

    shuffled_images = tf.gather(images, indices)
    mixed_images = mask * images + (1.0 - mask) * shuffled_images

    # Adjust lambda based on actual cut area
    actual_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    total_area = tf.cast(img_w * img_h, tf.float32)
    actual_lambda = 1.0 - actual_area / total_area
    actual_lambda = tf.expand_dims(actual_lambda, axis=1)

    shuffled_labels = tf.gather(labels, indices)
    mixed_labels = actual_lambda * labels + (1.0 - actual_lambda) * shuffled_labels

    return mixed_images, mixed_labels


# ── Data Pipeline ──────────────────────────────────────────────────────

def build_data_pipeline(data_dir: str, batch_size: int, img_size: int,
                        use_mixup: bool = True, is_training: bool = True):
    """
    Build train/val datasets from a directory of breed subfolders.

    Args:
        data_dir: Path to dataset with breed subfolders.
        batch_size: Batch size.
        img_size: Target image size (square).
        use_mixup: Whether to apply MixUp/CutMix on training data.
        is_training: If False, skips augmentation and shuffle.

    Returns:
        train_ds, val_ds, class_names, num_classes
    """
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="both",
        seed=42,
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

    # Rescale to [0, 1]
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)

    train_ds = train_ds.map(lambda x, y: (normalization(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Data augmentation
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

        # MixUp / CutMix
        if use_mixup:
            # Randomly pick MixUp or CutMix for each batch
            def apply_mixup(images, labels):
                if tf.random.uniform([]) < 0.5:
                    return mixup_batch(images, labels)
                else:
                    return cutmix_batch(images, labels)
            train_ds = train_ds.map(apply_mixup,
                                    num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


# ── Model Building ─────────────────────────────────────────────────────

def build_model(num_classes: int, backbone_name: str = "mobilenetv2",
                from_checkpoint: str = None, img_size: int = 224):
    """
    Build the model with the specified backbone.

    Args:
        num_classes: Number of dog breeds.
        backbone_name: Key in BACKBONES dict.
        from_checkpoint: Path to existing .h5 model.
        img_size: Input image size.

    Returns:
        model, base_model (the backbone for later unfreezing)
    """
    if from_checkpoint and os.path.exists(from_checkpoint):
        print(f"  Loading existing model from {from_checkpoint}...")
        import tensorflow_hub as hub
        model = tf.keras.models.load_model(
            from_checkpoint,
            custom_objects={"KerasLayer": hub.KerasLayer},
        )
        base_model = model.layers[1] if len(model.layers) > 1 else model.layers[0]
        return model, base_model

    cfg = BACKBONES[backbone_name]
    backbone_fn = cfg["fn"]

    base_model = backbone_fn(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Start frozen

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def unfreeze_backbone(base_model, unfreeze_from_layer: int):
    """Unfreeze the top layers of the backbone for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    total = len(base_model.layers)
    print(f"  Backbone: {trainable}/{total} layers trainable")


# ── Temperature Scaling ──────────────────────────────────────────────────

def fit_temperature_scaling(model, val_ds):
    """
    Fit a temperature T to calibrate softmax confidences.
    Minimizes NLL on the validation set using grid search over T.

    Works by converting model softmax probs back to log-probs (pseudo-logits),
    then dividing by T and re-softmaxing.  Because softmax is shift-invariant,
    log(probs) differs from true logits by a constant — which cancels out.

    Returns the optimal temperature (float > 0).
    """
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


# ── Training ───────────────────────────────────────────────────────────

def train(args):
    backbone_cfg = BACKBONES[args.backbone]
    use_mixup = args.mixup and not args.no_mixup

    # ── Phase 1: Train on 224x224 ──────────────────────────────────
    img_size = backbone_cfg["size"]
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training on {img_size}x{img_size}")
    print(f"  Backbone: {args.backbone}")
    print(f"  MixUp/CutMix: {use_mixup}")
    print(f"  Progressive resizing: {args.progressive}")
    print(f"{'='*60}")

    train_ds, val_ds, class_names, num_classes = build_data_pipeline(
        args.data_dir, args.batch_size, img_size, use_mixup=use_mixup
    )

    # Save class names
    os.makedirs("data", exist_ok=True)
    labels_path = "data/unique_breeds.json"
    with open(labels_path, "w") as f:
        json.dump(class_names, f)
    print(f"  Saved {len(class_names)} breed labels to {labels_path}")

    # Build model
    model, base_model = build_model(
        num_classes, args.backbone, args.resume, img_size
    )
    model.summary()

    # ── Step A: Train classifier head (backbone frozen) ────────────
    print(f"\n{'='*60}")
    print("STEP A: Training classifier head (backbone frozen)")
    print(f"{'='*60}")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger("training_log.csv"),
    ]

    head_epochs = max(5, args.epochs // 4)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=head_epochs, callbacks=callbacks, verbose=1,
    )

    # ── Step B: Fine-tune (unfreeze backbone) ──────────────────────
    print(f"\n{'='*60}")
    print("STEP B: Fine-tuning (unfreezing backbone layers)")
    print(f"{'='*60}")

    unfreeze_backbone(base_model, backbone_cfg["unfreeze_at"])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=backbone_cfg["lr"], weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    remaining_epochs = args.epochs - head_epochs
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=remaining_epochs,
        callbacks=callbacks + [
            tf.keras.callbacks.ModelCheckpoint(
                args.output, monitor="val_accuracy",
                save_best_only=True, verbose=1),
        ],
        verbose=1,
    )

    # ── Phase 2: Progressive resizing to 384x384 (optional) ────────
    if args.progressive:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Progressive resizing to 384x384")
        print(f"{'='*60}")

        # Rebuild data pipeline at larger size
        train_ds_384, val_ds_384, _, _ = build_data_pipeline(
            args.data_dir, args.batch_size // 2, 384, use_mixup=use_mixup
        )

        # Rebuild model with larger input
        model_384, base_model_384 = build_model(
            num_classes, args.backbone, img_size=384
        )

        # Transfer weights from 224 model
        # The backbone and dense layers are compatible — we just need to copy
        # the dense/classification weights
        for layer_224 in model.layers:
            for layer_384 in model_384.layers:
                if layer_224.name == layer_384.name and layer_224.weights:
                    layer_384.set_weights(layer_224.get_weights())
                    break

        model = model_384
        base_model = base_model_384

        # Fine-tune at 384
        unfreeze_backbone(base_model, backbone_cfg["unfreeze_at"])

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=backbone_cfg["lr"] / 2, weight_decay=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy",
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
                     tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
        )

        model.fit(
            train_ds_384, validation_data=val_ds_384,
            epochs=args.epochs // 3,
            callbacks=callbacks + [
                tf.keras.callbacks.ModelCheckpoint(
                    args.output.replace(".h5", "_384.h5"),
                    monitor="val_accuracy", save_best_only=True, verbose=1),
            ],
            verbose=1,
        )

    # ── Save & Evaluate ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"  Model saved to {args.output}")

    # Evaluate
    val_ds_final = val_ds_384 if args.progressive else val_ds
    val_loss, val_acc, val_top3, val_top5 = model.evaluate(val_ds_final, verbose=0)
    print(f"  Validation Accuracy:   {val_acc:.2%}")
    print(f"  Validation Top-3 Acc:  {val_top3:.2%}")
    print(f"  Validation Top-5 Acc:  {val_top5:.2%}")
    print(f"  Validation Loss:       {val_loss:.4f}")

    # ── Temperature scaling (confidence calibration) ─────────────
    print(f"\n{'='*60}")
    print("TEMPERATURE SCALING (Confidence Calibration)")
    print(f"{'='*60}")

    temperature = fit_temperature_scaling(model, val_ds_final)

    temp_path = os.path.join(os.path.dirname(args.output) or ".", "temp_scale.json")
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f)
    print(f"  Optimal temperature: {temperature:.4f}")
    print(f"  Saved to {temp_path}")

    return model, val_acc, val_top3, val_top5


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"\n❌ Data directory not found: {args.data_dir}")
        print("\nPlease download the Stanford Dogs Dataset:")
        print("  1. Visit: http://vision.stanford.edu/aditya86/ImageNetDogs/")
        print("  2. Extract images into data/Images/ (organized by breed folder)")
        print("  3. Or use --data_dir to point to your dataset")
        print("\n  Example: python train.py --data_dir data/Images --backbone efficientnetv2 --progressive")
        sys.exit(1)

    train(args)