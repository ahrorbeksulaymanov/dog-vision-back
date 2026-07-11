# --- Re-evaluate saved 224 models, robustly ---
# EfficientNet loads fine via load_model(). ConvNeXt's functional graph won't
# round-trip through Keras 3's legacy .h5 loader, so we fall back to rebuilding
# the exact architecture and loading weights BY POSITION. That position order
# depends on which layers were trainable at save time, so we reproduce the same
# partial unfreeze (last 30%) before load_weights — otherwise the weight list is
# reordered and shapes mismatch. Each model is then re-saved as a clean .keras.
import tensorflow as tf

DATA_DIR = "/kaggle/input/datasets/mozzamshahid/stanford-dogs-dataset/data/Images"
BATCH_SIZE = 32
NUM_CLASSES = 120

BACKBONES = {
    "efficientnetv2s": (tf.keras.applications.EfficientNetV2S, 0.3),
    "convnext": (tf.keras.applications.ConvNeXtTiny, 0.3),
}


def _get_layer_scale():
    for modpath in (
        "keras.src.applications.convnext",
        "keras.applications.convnext",
        "tensorflow.keras.applications.convnext",
    ):
        try:
            mod = __import__(modpath, fromlist=["LayerScale"])
        except ImportError:
            continue
        if hasattr(mod, "LayerScale"):
            return mod.LayerScale
    return None


def build_model(num_classes, backbone_name, img_size=224):
    # Must match build_model() in kaggle_train.py exactly.
    fn, _ = BACKBONES[backbone_name]
    base_model = fn(input_shape=(img_size, img_size, 3), include_top=False, weights=None)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base_model


def unfreeze_backbone(base_model, frac):
    # Reproduce the save-time trainable partition so weights load in order.
    base_model.trainable = True
    total = len(base_model.layers)
    freeze_until = int(total * (1.0 - frac))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False


def load_model_robust(name, path):
    custom_objects = {}
    ls = _get_layer_scale()
    if ls is not None:
        custom_objects["LayerScale"] = ls
    try:
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print("  loaded via load_model()")
        return model
    except Exception as e:
        print(f"  load_model() failed ({type(e).__name__}); rebuilding + load_weights")
        model, base = build_model(NUM_CLASSES, name)
        unfreeze_backbone(base, BACKBONES[name][1])  # match save-time weight order
        model.load_weights(path)
        print("  loaded via rebuild + load_weights()")
        return model


# Rebuild the val split the same way build_data_pipeline() does.
_, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
)
normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

METRICS = [
    "accuracy",
    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
]

for name, path in [
    ("efficientnetv2s", "/kaggle/working/dog_model_improved.h5"),
    ("convnext", "/kaggle/working/dog_model_convnext.h5"),
]:
    print(f"\n--- Re-evaluating {name} from {path} ---")
    model = load_model_robust(name, path)
    model.compile(loss="categorical_crossentropy", metrics=METRICS)
    val_loss, val_acc, val_top3, val_top5 = model.evaluate(val_ds, verbose=0)
    print(f"  [{name}] Validation Accuracy:  {val_acc:.2%}")
    print(f"  [{name}] Validation Top-3 Acc: {val_top3:.2%}")
    print(f"  [{name}] Validation Top-5 Acc: {val_top5:.2%}")

    keras_path = path.replace(".h5", ".keras")
    model.save(keras_path)
    print(f"  Saved clean copy to {keras_path}")
