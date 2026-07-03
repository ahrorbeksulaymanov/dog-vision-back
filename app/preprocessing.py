from io import BytesIO

import numpy as np
from PIL import Image

IMG_SIZE = 224


def preprocess_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image, dtype=np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image