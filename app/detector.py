import numpy as np
import tensorflow as tf

COCO_DOG_CLASS = 18


class DogDetector:
    """Dog detector using pre-trained EfficientDet-Lite0 from TensorFlow Hub.

    EfficientDet-Lite0 is a lightweight object detection model (~5MB) trained on
    the COCO dataset. It detects 80 object classes including "dog" (class 18,
    1-indexed). The model returns bounding boxes in **absolute pixel coordinates**
    (NOT normalized [0,1]) in [ymin, xmin, ymax, xmax] order.
    """

    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import tensorflow_hub as hub

            self._model = hub.load(
                "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1"
            )
        return self._model

    def detect(self, image: np.ndarray):
        """Detect dogs in a [H, W, 3] uint8 image.

        Returns list of dicts: {"bbox": [x1, y1, x2, y2], "confidence": float}
        Coordinates are **absolute pixel values** in the input image's space.
        """
        h, w = image.shape[:2]
        input_tensor = tf.convert_to_tensor(image[tf.newaxis, ...])
        output = self.model(input_tensor)

        if isinstance(output, dict):
            boxes = output["detection_boxes"].numpy()[0]
            classes = output["detection_classes"].numpy()[0].astype(int)
            scores = output["detection_scores"].numpy()[0]
            num = int(output["num_detections"].numpy()[0])
        else:
            boxes_out, scores_out, classes_out, num_t = output
            boxes = boxes_out.numpy()[0]
            scores = scores_out.numpy()[0]
            classes = classes_out.numpy()[0].astype(int)
            num = int(num_t.numpy()[0])

        detections = []
        for i in range(num):
            if classes[i] != COCO_DOG_CLASS:
                continue
            if scores[i] < self.confidence_threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            xmin = max(0.0, min(float(xmin), float(w)))
            ymin = max(0.0, min(float(ymin), float(h)))
            xmax = max(0.0, min(float(xmax), float(w)))
            ymax = max(0.0, min(float(ymax), float(h)))

            if xmax <= xmin or ymax <= ymin:
                continue

            detections.append(
                {
                    "bbox": [
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                    ],
                    "confidence": float(scores[i]),
                }
            )
        return detections


detector = DogDetector()