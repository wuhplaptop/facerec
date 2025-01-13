# detectors.py

import abc
from typing import List, Tuple

class FaceDetector(abc.ABC):
    """
    Abstract base class for any face detector.
    """
    @abc.abstractmethod
    def detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of (x1,y1,x2,y2) bounding boxes.
        """
        pass


class YOLOFaceDetector(FaceDetector):
    def __init__(self, model, conf_threshold=0.5):
        """
        Args:
            model: A loaded YOLO model (from ultralytics or similar).
            conf_threshold (float): confidence threshold for face detection.
        """
        self.model = model
        self.conf_threshold = conf_threshold

    def detect_faces(self, image):
        """
        Runs YOLO on the given image, returns bounding boxes.
        """
        results = self.model(image)
        boxes = []
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= self.conf_threshold and cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
