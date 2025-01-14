# myfacerec/facial_recognition.py

import os
import requests
import logging
import numpy as np
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
from typing import List, Optional

from .config import Config, logger
from .detectors import YOLOFaceDetector, FaceDetector
from .embedders import FacenetEmbedder, FaceEmbedder
from .data_store import JSONUserDataStore, UserDataStore
from .hooks import Hooks
from .plugins.base import PluginManager

class FacialRecognition:
    """
    Main orchestrator that uses:
      - a FaceDetector plugin
      - a FaceEmbedder plugin
      - a UserDataStore plugin
      - config hooks
    """

    def __init__(
        self,
        config: Config,
        detector: FaceDetector = None,
        embedder: FaceEmbedder = None,
        data_store: UserDataStore = None
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hooks = Hooks()

        # Initialize Plugin Manager
        self.plugin_manager = PluginManager()

        # If plugins are specified, load them
        if config.detector_plugin:
            detector = self.plugin_manager.load_detector(config.detector_plugin)
        if config.embedder_plugin:
            embedder = self.plugin_manager.load_embedder(config.embedder_plugin)

        # If no detector, set up YOLO by default
        if detector is None:
            if self.config.yolo_model_path:
                # Use custom model path (local or URL)
                path = self.config.yolo_model_path
                if path.startswith("http"):
                    # If it's a URL, download it locally
                    path = self._download_custom_model(path)
            else:
                # Use the default or download if missing
                path = self._download_model_if_needed(self.config.default_model_url)

            yolo_model = YOLO(path)
            yolo_model.to(config.device)
            detector = YOLOFaceDetector(yolo_model, conf_threshold=config.conf_threshold)
        self.detector = detector

        # If no embedder, use Facenet
        if embedder is None:
            fn_model = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)
            embedder = FacenetEmbedder(fn_model, device=config.device, alignment_fn=config.alignment_fn)
        self.embedder = embedder

        # If no data store, use JSON
        if data_store is None:
            data_store = JSONUserDataStore(config.user_data_path)
        self.data_store = data_store

        # Load user data
        self.user_data = self.data_store.load_user_data()
        self.logger.info("Initialized with %d users in data store.", len(self.user_data))

    def _download_model_if_needed(self, url):
        """
        Download the default YOLO model if not present.
        """
        base_dir = os.path.join(self.config.cache_dir, "models")
        os.makedirs(base_dir, exist_ok=True)
        model_path = os.path.join(base_dir, "face.pt")

        if not os.path.exists(model_path):
            self.logger.info("Downloading default YOLO model from %s", url)
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info("Default model saved to %s", model_path)

        return model_path

    def _download_custom_model(self, url):
        """
        Download a custom YOLO model from the given URL to a local path.
        """
        base_dir = os.path.join(self.config.cache_dir, "custom_models")
        os.makedirs(base_dir, exist_ok=True)
        filename = os.path.basename(url)
        model_path = os.path.join(base_dir, filename)

        if not os.path.exists(model_path):
            self.logger.info("Downloading custom YOLO model from %s", url)
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info("Custom model saved to %s", model_path)

        return model_path

    def detect_faces(self, image):
        if self.hooks.before_detect:
            image = self.hooks.before_detect(image)

        boxes = self.detector.detect_faces(image)

        if self.hooks.after_detect:
            self.hooks.after_detect(boxes)

        return boxes

    def embed_faces_batch(self, image, boxes):
        if self.hooks.before_embed:
            self.hooks.before_embed(image, boxes)

        embeddings = self.embedder.embed_faces_batch(image, boxes)

        if self.hooks.after_embed:
            self.hooks.after_embed(embeddings)

        return embeddings

    def register_user(self, user_id: str, images: List[Image.Image]):
        collected_embeddings = []
        for img in images:
            boxes = self.detect_faces(img)
            # Only register if there's exactly one face in the image
            if len(boxes) == 1:
                emb = self.embed_faces_batch(img, boxes)
                if emb.shape[0] == 1:
                    collected_embeddings.append(emb[0])

        if not collected_embeddings:
            return f"[Registration] No valid single-face images found for '{user_id}'."

        if user_id not in self.user_data:
            self.user_data[user_id] = []
        self.user_data[user_id].extend(collected_embeddings)

        self.data_store.save_user_data(self.user_data)
        return f"[Registration] User '{user_id}' registered with {len(collected_embeddings)} images."

    def identify_user(self, image: Image.Image, threshold=0.6):
        boxes = self.detect_faces(image)
        if not boxes:
            return []

        embeddings = self.embed_faces_batch(image, boxes)
        results = []

        for i, emb in enumerate(embeddings):
            best_match = None
            best_sim = 0.0

            for user_id, emb_list in self.user_data.items():
                valid_embs = [e for e in emb_list if e.shape == emb.shape]
                if not valid_embs:
                    continue
                sims = cosine_similarity([emb], valid_embs)
                max_sim = np.max(sims)
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_match = user_id

            if best_sim >= threshold:
                results.append({
                    'box': boxes[i],
                    'user_id': best_match,
                    'similarity': float(best_sim)
                })
            else:
                results.append({
                    'box': boxes[i],
                    'user_id': 'Unknown',
                    'similarity': float(best_sim)
                })

        return results

    def list_users(self):
        return list(self.user_data.keys())

    def delete_user(self, user_id: str):
        if user_id in self.user_data:
            del self.user_data[user_id]
            self.data_store.save_user_data(self.user_data)
            return True
        return False

    def update_user_embeddings(self, user_id: str, new_embeddings: List[np.ndarray]):
        """
        Add new embeddings to an existing user.
        """
        if user_id not in self.user_data:
            return False
        self.user_data[user_id].extend(new_embeddings)
        self.data_store.save_user_data(self.user_data)
        return True

    def retrain_models(self):
        """
        Placeholder for continuous learning logic.
        Implement retraining or model updates as needed.
        """
        # Example: Recompute average embeddings or retrain embedder
        pass
