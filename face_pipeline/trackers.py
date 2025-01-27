# face_pipeline/trackers.py

import logging
import numpy as np
from typing import List, Tuple
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, max_age: int = 30):
        self.tracker = DeepSort(max_age=max_age, embedder='mobilenet')

    def update(self, detections: List[Tuple], frame: np.ndarray):
        try:
            ds_detections = [
                ([x1, y1, x2 - x1, y2 - y1], conf, cls)
                for (x1, y1, x2, y2, conf, cls) in detections
            ]
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
            logger.debug(f"Updated tracker with {len(tracks)} tracks.")
            return tracks
        except Exception as e:
            logger.error(f"Tracking update failed: {str(e)}")
            return []
