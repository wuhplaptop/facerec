# face_pipeline/detectors.py

import os
import requests
import logging
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://github.com/wuhplaptop/face-11-n/blob/main/face2.pt?raw=true"

class YOLOFaceDetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = None
        self.device = device

        try:
            if not os.path.exists(model_path):
                logger.info(f"Model file not found at {model_path}. Attempting to download...")
                response = requests.get(DEFAULT_MODEL_URL)
                response.raise_for_status()
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded YOLO model to {model_path}")

            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"YOLO initialization failed: {str(e)}")
            raise

    def detect(self, image: np.ndarray, conf_thres: float) -> List[Tuple[int, int, int, int, float, int]]:
        try:
            results = self.model.predict(
                source=image, conf=conf_thres, verbose=False, device=self.device
            )

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
            logger.debug(f"Detected {len(detections)} faces.")
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return []
