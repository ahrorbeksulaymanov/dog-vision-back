from io import BytesIO

import numpy as np
from PIL import Image

IMG_SIZE = 224


def preprocess_image(image_bytes: bytes, img_size: int = IMG_SIZE):
    """Preprocess a single image for prediction."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_tta(image_bytes: bytes, num_augments: int = 8,
                         img_size: int = IMG_SIZE):
    """
    Preprocess an image with test-time augmentation (TTA) variants.

    Returns a batch of augmented images: original + flipped/rotated versions.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Resize slightly larger for cropping
    pad = 32
    image = image.resize((img_size + pad, img_size + pad))

    variants = []

    # Original (center crop)
    orig = _center_crop(image, img_size)
    variants.append(np.array(orig, dtype=np.float32))

    # Horizontal flip
    flipped = orig.transpose(Image.FLIP_LEFT_RIGHT)
    variants.append(np.array(flipped, dtype=np.float32))

    # Corner crops
    variants.append(np.array(image.crop((0, 0, img_size, img_size)), dtype=np.float32))
    variants.append(np.array(image.crop((pad, 0, pad + img_size, img_size)), dtype=np.float32))
    variants.append(np.array(image.crop((0, pad, img_size, pad + img_size)), dtype=np.float32))
    variants.append(np.array(image.crop((pad, pad, pad + img_size, pad + img_size)), dtype=np.float32))

    # Flip the corner crops too
    for i in range(2, 6):
        flipped_crop = Image.fromarray(variants[i].astype(np.uint8)).transpose(Image.FLIP_LEFT_RIGHT)
        variants.append(np.array(flipped_crop, dtype=np.float32))

    # Limit to requested number
    variants = variants[:num_augments]

    # Normalize and batch
    batch = np.stack(variants, axis=0) / 255.0
    return batch


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    """Center crop an image to `size x size`."""
    w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))
