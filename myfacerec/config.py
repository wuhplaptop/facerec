# myfacerec/config.py

import logging
import os
import pkg_resources
import requests

class Config:
    """
    Central configuration object.
    """
    def __init__(
        self,
        yolo_model_path=None,
        default_model_url="https://github.com/wuhplaptop/facerec/raw/main/myfacerec/models/face.pt",
        conf_threshold=0.5,
        similarity_threshold=0.6,
        device=None,
        user_data_path="user_faces.json",
        alignment_fn=None, 
        before_detect=None,
        after_detect=None,
        before_embed=None,
        after_embed=None,
        detector_plugin=None,
        embedder_plugin=None,
        cache_dir=None,
        enable_pose_estimation=False  # NEW
    ):
        self.yolo_model_path = yolo_model_path or self._default_model_path(default_model_url)
        self.default_model_url = default_model_url
        self.conf_threshold = conf_threshold
        self.similarity_threshold = similarity_threshold
        self.device = device or self._auto_device()
        self.user_data_path = user_data_path

        # Optional hooks
        self.alignment_fn = alignment_fn
        self.before_detect = before_detect
        self.after_detect = after_detect
        self.before_embed = before_embed
        self.after_embed = after_embed

        # Plugins
        self.detector_plugin = detector_plugin
        self.embedder_plugin = embedder_plugin

        # Caching
        self.cache_dir = cache_dir or os.path.expanduser("~/.myfacerec/cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # NEW: pose estimation flag
        self.enable_pose_estimation = enable_pose_estimation

        # Ensure YOLO model is present
        if not os.path.exists(self.yolo_model_path):
            self._download_default_yolo_model()

    def _auto_device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _default_model_path(self, default_model_url):
        local_model_path = pkg_resources.resource_filename(__name__, 'models/face.pt')
        if not os.path.exists(local_model_path):
            self._download_model(default_model_url, local_model_path)
        return local_model_path

    def _download_model(self, url, save_path):
        try:
            logging.info(f"Downloading default YOLO model from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.wr
