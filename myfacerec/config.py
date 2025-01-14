# myfacerec/config.py

import logging
import os

class Config:
    """
    Central configuration object.
    """
    def __init__(
        self,
        yolo_model_path=None,
        default_model_url="https://raw.githubusercontent.com/wuhplaptop/facerec/main/face.pt",
        conf_threshold=0.5,
        device=None,
        user_data_path="user_faces.json",
        alignment_fn=None,
        before_detect=None,
        after_detect=None,
        before_embed=None,
        after_embed=None,
        detector_plugin=None,
        embedder_plugin=None,
        cache_dir=None
    ):
        self.yolo_model_path = yolo_model_path
        self.default_model_url = default_model_url
        self.conf_threshold = conf_threshold
        self.device = device or self._auto_device()
        self.user_data_path = user_data_path

        # Optional hooks
        self.before_detect = before_detect
        self.after_detect = after_detect
        self.before_embed = before_embed
        self.after_embed = after_embed

        # Plugins
        self.detector_plugin = detector_plugin
        self.embedder_plugin = embedder_plugin

        # Caching
        self.cache_dir = cache_dir or os.path.expanduser("~/.myfacerec/cache")

    def _auto_device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


# Basic logger setup (global)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
