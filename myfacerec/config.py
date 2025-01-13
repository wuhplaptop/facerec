# config.py

import os
import logging

class Config:
    """
    Central configuration object. Instead of passing multiple
    arguments everywhere, define them here.
    """
    def __init__(self,
                 yolo_model_path=None,
                 default_model_url="https://raw.githubusercontent.com/wuhplaptop/facerec/main/face.pt",
                 conf_threshold=0.5,
                 device=None,
                 user_data_path="user_faces.json",
                 alignment_fn=None,
                 before_detect=None,
                 after_detect=None,
                 before_embed=None,
                 after_embed=None):
        """
        Args:
            yolo_model_path (str): Path to YOLO .pt. If None, will download from default_model_url.
            default_model_url (str): Where to fetch default YOLO model if needed.
            conf_threshold (float): YOLO detection threshold.
            device (str): "cpu" or "cuda". If None, auto-detects.
            user_data_path (str): Where to store embeddings.
            alignment_fn (callable): Optional face alignment/preprocessing function.
            before_detect (callable): Hook called before detection step.
            after_detect (callable): Hook called after detection step.
            before_embed (callable): Hook called before embedding.
            after_embed (callable): Hook called after embedding.
        """
        self.yolo_model_path = yolo_model_path
        self.default_model_url = default_model_url
        self.conf_threshold = conf_threshold
        self.device = device or self._auto_device()
        self.user_data_path = user_data_path

        # Hooks / alignment
        self.alignment_fn = alignment_fn
        self.before_detect = before_detect
        self.after_detect = after_detect
        self.before_embed = before_embed
        self.after_embed = after_embed

    def _auto_device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging (optional):
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
