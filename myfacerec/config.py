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
        similarity_threshold=0.6,  # Added similarity_threshold
        device=None,
        user_data_path="user_faces.json",
        alignment_fn=None,     # Optional face alignment function
        before_detect=None,
        after_detect=None,
        before_embed=None,
        after_embed=None,
        detector_plugin=None,
        embedder_plugin=None,
        cache_dir=None
    ):
        self.yolo_model_path = yolo_model_path or self._default_model_path(default_model_url)
        self.default_model_url = default_model_url
        self.conf_threshold = conf_threshold
        self.similarity_threshold = similarity_threshold  # Assign it
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

        # Ensure YOLO model is present
        if not os.path.exists(self.yolo_model_path):
            self._download_default_yolo_model()

    def _auto_device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _default_model_path(self, default_model_url):
        """
        Returns the default model path, which is 'models/face.pt' within the package.
        Downloads the model if it doesn't exist.
        """
        local_model_path = pkg_resources.resource_filename(__name__, 'models/face.pt')
        if not os.path.exists(local_model_path):
            self._download_model(default_model_url, local_model_path)
        return local_model_path

    def _download_model(self, url, save_path):
        """
        Downloads a model from a URL and saves it locally.
        """
        try:
            logging.info(f"Downloading default YOLO model from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Default YOLO model downloaded and saved to {save_path}.")
        except Exception as e:
            logging.error(f"Failed to download YOLO model from {url}: {e}")
            raise

    def _download_default_yolo_model(self):
        """
        Downloads the default YOLO model if it's not present.
        """
        os.makedirs(os.path.dirname(self.yolo_model_path), exist_ok=True)
        self._download_model(self.default_model_url, self.yolo_model_path)

# Basic logger setup (global)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
