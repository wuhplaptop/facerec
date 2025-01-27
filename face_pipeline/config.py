# face_pipeline/config.py

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    detector: Dict = field(default_factory=dict)
    tracker: Dict = field(default_factory=dict)
    recognition: Dict = field(default_factory=dict)
    anti_spoof: Dict = field(default_factory=dict)
    blink: Dict = field(default_factory=dict)
    face_mesh_options: Dict = field(default_factory=dict)
    hand: Dict = field(default_factory=dict)
    eye_color: Dict = field(default_factory=dict)
    enabled_components: Dict = field(default_factory=dict)
    detection_conf_thres: float = 0.4
    recognition_conf_thres: float = 0.85
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    spoofed_bbox_color: Tuple[int, int, int] = (0, 0, 255)
    unknown_bbox_color: Tuple[int, int, int] = (0, 0, 255)
    eye_outline_color: Tuple[int, int, int] = (255, 255, 0)
    blink_text_color: Tuple[int, int, int] = (0, 0, 255)
    hand_landmark_color: Tuple[int, int, int] = (255, 210, 77)
    hand_connection_color: Tuple[int, int, int] = (204, 102, 0)
    hand_text_color: Tuple[int, int, int] = (255, 255, 255)
    mesh_color: Tuple[int, int, int] = (100, 255, 100)
    contour_color: Tuple[int, int, int] = (200, 200, 0)
    iris_color: Tuple[int, int, int] = (255, 0, 255)
    eye_color_text_color: Tuple[int, int, int] = (255, 255, 255)

    DEFAULT_DB_PATH = os.path.expanduser("~/.face_pipeline/known_faces.pkl")
    MODEL_DIR = os.path.expanduser("~/.face_pipeline/models")
    CONFIG_PATH = os.path.expanduser("~/.face_pipeline/config.pkl")

    def __post_init__(self):
        self.detector = self.detector or {
            'model_path': os.path.join(self.MODEL_DIR, "face2.pt"),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        self.tracker = self.tracker or {'max_age': 30}
        self.recognition = self.recognition or {'enable': True}
        self.anti_spoof = self.anti_spoof or {'enable': True, 'lap_thresh': 80.0}
        self.blink = self.blink or {'enable': True, 'ear_thresh': 0.25}
        self.face_mesh_options = self.face_mesh_options or {
            'enable': False,
            'tesselation': False,
            'contours': False,
            'irises': False,
        }
        self.hand = self.hand or {
            'enable': True,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
        }
        self.eye_color = self.eye_color or {'enable': False}
        self.enabled_components = self.enabled_components or {
            'detection': True,
            'tracking': True,
            'anti_spoof': True,
            'recognition': True,
            'blink': True,
            'face_mesh': False,
            'hand': True,
            'eye_color': False,
        }

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.__dict__, f)
            logger.info(f"Saved config to {path}")
        except Exception as e:
            logger.error(f"Config save failed: {str(e)}")
            raise RuntimeError(f"Config save failed: {str(e)}") from e

    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                return cls(**data)
            return cls()
        except Exception as e:
            logger.error(f"Config load failed: {str(e)}")
            return cls()
