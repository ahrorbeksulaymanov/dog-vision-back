import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from app.preprocessing import preprocess_image


class DogBreedPredictor:

    def __init__(self):
        self.model = tf.keras.models.load_model(
            "models/dog_model.h5",
            custom_objects={"KerasLayer": hub.KerasLayer},
        )

        with open("data/unique_breeds.json", "r") as f:
            self.labels = json.load(f)

    def predict(self, image_bytes: bytes):

        image = preprocess_image(image_bytes)

        predictions = self.model.predict(image, verbose=0)

        index = np.argmax(predictions)

        confidence = float(predictions[0][index])

        breed = self.labels[index]

        return {
            "breed": breed,
            "confidence": confidence,
        }


predictor = DogBreedPredictor()