import time
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from app.detector import detector
from app.predictor import predictor


class LivePipeline:
    """Live detection pipeline: EfficientDet-Lite0 → single-model breed classification.

    Uses only the smallest (fastest) breed model for live mode instead of the
    full ensemble. This roughly halves inference time per frame. All detected
    dog crops are classified in a single batched forward pass for efficiency.
    """

    def __init__(self):
        self.prev_time: Optional[float] = None
        self._fast_idx: Optional[int] = None

    @property
    def fast_idx(self) -> int:
        if self._fast_idx is None:
            self._fast_idx = min(
                range(len(predictor.models)),
                key=lambda i: predictor.models[i].count_params(),
            )
            params = predictor.models[self._fast_idx].count_params()
            print(
                f"[pipeline] Fast model: index {self._fast_idx} "
                f"({params:,} params) — single-model mode for live"
            )
        return self._fast_idx

    def _classify_crops(self, crops: list) -> list:
        """Batch-classify dog crops using the fast single model."""
        if not crops:
            return []

        model = predictor.models[self.fast_idx]
        temperature = predictor.temperatures[self.fast_idx]
        img_size = int(model.input_shape[1] or 224)

        batch = np.stack(
            [
                np.array(c.resize((img_size, img_size)), dtype=np.float32)
                / 255.0
                for c in crops
            ],
            axis=0,
        )

        preds = model.predict(batch, verbose=0)
        preds = predictor._apply_temperature(preds, temperature)

        results = []
        for i in range(len(crops)):
            top_idx = int(np.argmax(preds[i]))
            confidence = float(preds[i][top_idx])
            breed = predictor.labels[top_idx]
            is_unknown = confidence < predictor.confidence_threshold
            results.append(
                {
                    "breed": breed if not is_unknown else "unknown",
                    "breed_conf": round(confidence, 4) if not is_unknown else 0.0,
                    "is_unknown": is_unknown,
                }
            )
        return results

    def process_frame(self, jpeg_bytes: bytes):
        image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        frame = np.array(image, dtype=np.uint8)
        h, w = frame.shape[:2]

        dog_detections = detector.detect(frame)

        crops = []
        validated = []
        for det in dog_detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            validated.append((det, x1, y1, x2, y2))

        breed_preds = self._classify_crops(crops)

        results = []
        for i, (det, x1, y1, x2, y2) in enumerate(validated):
            bp = breed_preds[i]
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": round(det["confidence"], 4),
                    "breed": bp["breed"],
                    "breed_conf": bp["breed_conf"],
                }
            )

        now = time.time()
        fps = 0.0
        if self.prev_time is not None:
            elapsed = now - self.prev_time
            if elapsed > 0:
                fps = round(1.0 / elapsed, 1)
        self.prev_time = now

        return {"detections": results, "fps": fps, "width": w, "height": h}


pipeline = LivePipeline()