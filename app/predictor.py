import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from app.preprocessing import preprocess_image, preprocess_image_tta


def _get_layer_scale():
    """Find ConvNeXt's LayerScale layer class, wherever this Keras version hides it.

    Its import path moved across Keras releases and isn't consistently exposed
    on tf.keras.applications.convnext, so an attribute check alone silently
    fails. Fall back to reconstructing the class from its known source so
    saved weights still deserialize.
    """
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

    from tensorflow import keras

    class LayerScale(keras.layers.Layer):
        def __init__(self, init_values, projection_dim, **kwargs):
            super().__init__(**kwargs)
            self.init_values = init_values
            self.projection_dim = projection_dim

        def build(self, _):
            self.gamma = self.add_weight(
                shape=(self.projection_dim,),
                initializer=keras.initializers.Constant(self.init_values),
                trainable=True,
                name="gamma",
            )

        def call(self, x):
            return x * self.gamma

        def get_config(self):
            config = super().get_config()
            config.update({"init_values": self.init_values, "projection_dim": self.projection_dim})
            return config

    return LayerScale


# Backbone config for rebuilding architectures when load_model() can't
# deserialize a saved graph (e.g. ConvNeXt through Keras 3's legacy .h5 loader).
# unfreeze_frac must match what training used so the by-position weight order
# lines up — see build_model()/unfreeze_backbone() in kaggle_train.py.
_BACKBONES = {
    "efficientnetv2s": (lambda **kw: tf.keras.applications.EfficientNetV2S(**kw), 0.3),
    "convnext": (lambda **kw: tf.keras.applications.ConvNeXtTiny(**kw), 0.3),
}


def _backbone_for_path(path: str) -> str:
    """Infer which backbone a saved model uses from its filename convention."""
    return "convnext" if "convnext" in os.path.basename(path).lower() else "efficientnetv2s"


def _build_model(num_classes: int, backbone_name: str, img_size: int = 224):
    """Rebuild the exact architecture from kaggle_train.py's build_model()."""
    backbone_fn, _ = _BACKBONES[backbone_name]
    base_model = backbone_fn(input_shape=(img_size, img_size, 3), include_top=False, weights=None)
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


def _reproduce_unfreeze(base_model, frac: float):
    """Reproduce training's partial unfreeze so load_weights() orders correctly.

    Legacy .h5 stores weights as trainable + non_trainable; the fine-tuned
    models were saved with the last `frac` of the backbone unfrozen, so a
    fully-frozen rebuild would load weights in the wrong order (shape mismatch).
    """
    base_model.trainable = True
    total = len(base_model.layers)
    freeze_until = int(total * (1.0 - frac))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False


def _load_model_robust(path: str, num_classes: int, custom_objects: dict):
    """Load a saved model, rebuilding + loading weights if deserialization fails.

    ConvNeXt's functional graph can't round-trip through Keras 3's legacy .h5
    loader, so fall back to reconstructing the architecture and loading only the
    weights (which we verified reproduces the exact validation accuracy).
    """
    try:
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    except Exception as e:
        backbone = _backbone_for_path(path)
        print(f"  load_model() failed for {path} ({type(e).__name__}); "
              f"rebuilding as {backbone} + load_weights")
        model, base_model = _build_model(num_classes, backbone)
        _reproduce_unfreeze(base_model, _BACKBONES[backbone][1])
        model.load_weights(path)
        return model


class DogBreedPredictor:

    def __init__(
        self,
        model_paths: list | str = "models/dog_model.h5",
        labels_path: str = "data/unique_breeds.json",
        confidence_threshold: float = 0.50,
        top_k: int = 3,
    ):
        """
        Initialize the predictor.

        Args:
            model_paths: Single model path or list of paths for ensemble.
            labels_path: Path to breed labels JSON.
            confidence_threshold: Below this, return "unknown".
            top_k: Number of top predictions to return.
        """
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        # Labels first: num_classes is needed to rebuild architectures when
        # load_model() falls back to reconstruction + load_weights.
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        num_classes = len(self.labels)

        custom_objects = {"KerasLayer": hub.KerasLayer, "LayerScale": _get_layer_scale()}

        self.models = []
        loaded_paths = []
        for path in model_paths:
            if not os.path.exists(path):
                print(f"Warning: Model not found at {path}, skipping.")
                continue
            try:
                model = _load_model_robust(path, num_classes, custom_objects)
            except Exception as e:
                print(f"Warning: Failed to load {path} ({e}), skipping.")
                continue
            self.models.append(model)
            loaded_paths.append(path)
        model_paths = loaded_paths

        if not self.models:
            raise FileNotFoundError(
                f"No models found. Checked: {model_paths}"
            )

        print(f"Loaded {len(self.models)} model(s):")
        for i, (m, p) in enumerate(zip(self.models, model_paths)):
            params = m.count_params()
            print(f"  [{i+1}] {p} ({params:,} params)")

        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.num_models = len(self.models)

        # Per-model temperature scaling (confidence calibration). Each model was
        # calibrated separately, so prefer its sibling <name>_temp_scale.json;
        # fall back to a shared temp_scale.json, then to T=1.0 (no scaling).
        self.temperatures = [self._load_temperature(p) for p in model_paths]
        for p, t in zip(model_paths, self.temperatures):
            state = f"T={t:.4f}" if t != 1.0 else "disabled (T=1.0)"
            print(f"  Temperature [{os.path.basename(p)}]: {state}")

    @staticmethod
    def _load_temperature(model_path: str) -> float:
        """Find the temperature for one model: sibling file, then shared, then 1.0."""
        models_dir = os.path.dirname(model_path) or "models"
        base = os.path.splitext(os.path.basename(model_path))[0]
        candidates = [
            os.path.join(models_dir, f"{base}_temp_scale.json"),
            os.path.join(models_dir, "temp_scale.json"),
        ]
        for temp_path in candidates:
            if os.path.exists(temp_path):
                with open(temp_path, "r") as f:
                    return json.load(f).get("temperature", 1.0)
        return 1.0

    def _apply_temperature(self, probs, temperature):
        """Apply temperature scaling to softmax probabilities."""
        if temperature == 1.0:
            return probs
        logits = np.log(probs + 1e-8)
        logits = logits / temperature
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        calibrated = np.exp(logits)
        calibrated = calibrated / np.sum(calibrated, axis=-1, keepdims=True)
        return calibrated

    def predict(self, image_bytes: bytes, use_tta: bool = False):
        """
        Predict the dog breed from image bytes.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG).
            use_tta: Whether to use test-time augmentation.

        Returns:
            dict with primary, top_k, is_unknown, and ensemble info.
        """
        # Get predictions from all models
        all_predictions = []
        for model, temperature in zip(self.models, self.temperatures):
            # Resize to whatever this model expects (224 or 384) instead of
            # assuming 224 — lets 384 checkpoints join the ensemble safely.
            img_size = int(model.input_shape[1] or 224)
            if use_tta:
                batch = preprocess_image_tta(image_bytes, img_size=img_size)
                preds = model.predict(batch, verbose=0)
                preds = np.mean(preds, axis=0)
            else:
                image = preprocess_image(image_bytes, img_size=img_size)
                preds = model.predict(image, verbose=0)[0]
            preds = self._apply_temperature(preds, temperature)
            all_predictions.append(preds)

        # Ensemble: average predictions across models
        if len(all_predictions) > 1:
            ensemble = np.mean(all_predictions, axis=0)
            # Calculate agreement between models
            individual_breeds = [
                self.labels[np.argmax(p)] for p in all_predictions
            ]
            agreement = len(set(individual_breeds)) == 1
        else:
            ensemble = all_predictions[0]
            individual_breeds = [self.labels[np.argmax(ensemble)]]
            agreement = True

        # Get top-K indices
        top_indices = np.argsort(ensemble)[-self.top_k:][::-1]

        top_predictions = []
        for rank, idx in enumerate(top_indices):
            confidence = float(ensemble[idx])
            breed = self.labels[idx]
            top_predictions.append({
                "rank": rank + 1,
                "breed": breed,
                "confidence": round(confidence, 4),
            })

        primary = top_predictions[0]
        is_unknown = primary["confidence"] < self.confidence_threshold

        result = {
            "primary": {
                "breed": primary["breed"] if not is_unknown else "unknown",
                "confidence": primary["confidence"],
            },
            "top_k": top_predictions,
            "is_unknown": is_unknown,
            "message": (
                "Confidence too low — breed may not be in our database."
                if is_unknown
                else "Prediction successful."
            ),
        }

        # Add ensemble metadata when multiple models are used
        if self.num_models > 1:
            result["ensemble"] = {
                "num_models": self.num_models,
                "all_agree": agreement,
                "individual_predictions": [
                    {"breed": self.labels[np.argmax(p)],
                     "confidence": round(float(np.max(p)), 4)}
                    for p in all_predictions
                ],
            }

        return result


# Auto-detect ensemble models in the models/ directory
def _find_models(models_dir: str = "models") -> list:
    """Find model files in the models directory, preferring .keras over .h5.

    The .keras files are the clean re-saved copies; if a model exists in both
    formats we keep only the .keras one so it isn't loaded (and ensembled) twice.
    """
    if not os.path.isdir(models_dir):
        return ["models/dog_model.h5"]

    all_files = os.listdir(models_dir)
    keras_stems = {
        os.path.splitext(f)[0] for f in all_files if f.endswith(".keras")
    }
    model_files = sorted(
        f for f in all_files
        if os.path.isfile(os.path.join(models_dir, f))
        and (
            f.endswith(".keras")
            or (f.endswith(".h5") and os.path.splitext(f)[0] not in keras_stems)
        )
    )

    if not model_files:
        return ["models/dog_model.h5"]

    return [os.path.join(models_dir, f) for f in model_files]


# Singleton instance
predictor = DogBreedPredictor(
    model_paths=_find_models(),
    confidence_threshold=0.50,
    top_k=3,
)