import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from app.preprocessing import preprocess_image, preprocess_image_tta


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

        self.models = []
        for path in model_paths:
            if not os.path.exists(path):
                print(f"Warning: Model not found at {path}, skipping.")
                continue
            model = tf.keras.models.load_model(
                path,
                custom_objects={"KerasLayer": hub.KerasLayer},
            )
            self.models.append(model)

        if not self.models:
            raise FileNotFoundError(
                f"No models found. Checked: {model_paths}"
            )

        print(f"Loaded {len(self.models)} model(s):")
        for i, (m, p) in enumerate(zip(self.models, model_paths)):
            params = m.count_params()
            print(f"  [{i+1}] {p} ({params:,} params)")

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.num_models = len(self.models)

        # Load temperature scaling (confidence calibration)
        self.temperature = 1.0
        models_dir = os.path.dirname(model_paths[0]) if model_paths else "models"
        temp_path = os.path.join(models_dir, "temp_scale.json")
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                self.temperature = json.load(f).get("temperature", 1.0)
            print(f"  Temperature scaling: T={self.temperature:.4f}")
        else:
            print(f"  Temperature scaling: disabled (T=1.0, no temp_scale.json)")

    def _apply_temperature(self, probs):
        """Apply temperature scaling to softmax probabilities."""
        if self.temperature == 1.0:
            return probs
        logits = np.log(probs + 1e-8)
        logits = logits / self.temperature
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
        for model in self.models:
            if use_tta:
                batch = preprocess_image_tta(image_bytes)
                preds = model.predict(batch, verbose=0)
                preds = np.mean(preds, axis=0)
            else:
                image = preprocess_image(image_bytes)
                preds = model.predict(image, verbose=0)[0]
            preds = self._apply_temperature(preds)
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
    """Find all .h5 model files in the models directory."""
    if not os.path.isdir(models_dir):
        return ["models/dog_model.h5"]

    model_files = sorted([
        f for f in os.listdir(models_dir)
        if f.endswith(".h5") and os.path.isfile(os.path.join(models_dir, f))
    ])

    if not model_files:
        return ["models/dog_model.h5"]

    return [os.path.join(models_dir, f) for f in model_files]


# Singleton instance
predictor = DogBreedPredictor(
    model_paths=_find_models(),
    confidence_threshold=0.50,
    top_k=3,
)