import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from app.detector import detector
from app.predictor import predictor

_MAX_WORKERS = 4


class LivePipeline:
    def __init__(self, max_workers: int = _MAX_WORKERS):
        self.prev_time: Optional[float] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _classify_crop(self, crop: Image.Image):
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        crop_bytes = buf.getvalue()
        try:
            return predictor.predict(crop_bytes, use_tta=False)
        except Exception:
            return None

    def process_frame(self, jpeg_bytes: bytes):
        image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        frame = np.array(image, dtype=np.uint8)
        h, w = frame.shape[:2]

        dog_detections = detector.detect(frame)

        crops = []
        validated_boxes = []
        for det in dog_detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
            validated_boxes.append((det, x1, y1, x2, y2))

        if crops:
            preds = list(self._executor.map(self._classify_crop, crops))
        else:
            preds = []

        results = []
        for i, (det, x1, y1, x2, y2) in enumerate(validated_boxes):
            pred = preds[i]
            result = {
                "bbox": [x1, y1, x2, y2],
                "det_conf": round(det["confidence"], 4),
            }
            if pred and not pred["is_unknown"]:
                result["breed"] = pred["primary"]["breed"]
                result["breed_conf"] = pred["primary"]["confidence"]
                result["top_3"] = pred["top_k"]
            else:
                result["breed"] = "unknown"
                result["breed_conf"] = 0.0
            results.append(result)

        now = time.time()
        fps = 0.0
        if self.prev_time is not None:
            elapsed = now - self.prev_time
            if elapsed > 0:
                fps = round(1.0 / elapsed, 1)
        self.prev_time = now

        return {"detections": results, "fps": fps, "width": w, "height": h}


pipeline = LivePipeline()