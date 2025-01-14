import os
import requests
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Dict, Any  # Added Dict and Any

from PIL import Image

from .config import Config, logger
from .detectors import YOLOFaceDetector, FaceDetector
from .embedders import FacenetEmbedder, FaceEmbedder
from .data_store import JSONUserDataStore, UserDataStore
from .hooks import Hooks
from .plugins.base import PluginManager
from .combined_model import CombinedFacialRecognitionModel

import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


class FacialRecognition:
    """
    Main orchestrator that uses:
      - a FaceDetector plugin or a combined model
      - a FaceEmbedder plugin (if not using combined model)
      - a UserDataStore plugin
      - config hooks
    """

    def __init__(
        self,
        config: Config,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        data_store: Optional[UserDataStore] = None,
        combined_model_path: Optional[str] = None  # New parameter for combined model
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hooks = Hooks()
        self.plugin_manager = PluginManager()

        # Initialize combined model if path is provided
        if combined_model_path:
            if not os.path.exists(combined_model_path):
                raise FileNotFoundError(f"Combined model file not found: {combined_model_path}")
            try:
                self.combined_model = CombinedFacialRecognitionModel.load_model(
                    combined_model_path, device=config.device
                )
                self.logger.info("Initialized with combined YOLO and Facenet model.")
            except Exception as e:
                self.logger.error(f"Failed to load combined model: {e}")
                raise
        else:
            self._initialize_separate_components(detector, embedder, data_store)

    def _initialize_separate_components(
        self,
        detector: Optional[FaceDetector],
        embedder: Optional[FaceEmbedder],
        data_store: Optional[UserDataStore]
    ):
        """Initialize individual components when combined model is not used."""
        if self.config.detector_plugin:
            detector = self.plugin_manager.load_detector(self.config.detector_plugin)
        if self.config.embedder_plugin:
            embedder = self.plugin_manager.load_embedder(self.config.embedder_plugin)

        if detector is None:
            detector = self._initialize_detector()
        self.detector = detector

        if embedder is None:
            embedder = self._initialize_embedder()
        self.embedder = embedder

        if data_store is None:
            data_store = JSONUserDataStore(self.config.user_data_path)
        self.data_store = data_store

        self.user_data = self.data_store.load_user_data()
        self.logger.info("Initialized with %d users in data store.", len(self.user_data))

    def _initialize_detector(self) -> YOLOFaceDetector:
        """Initialize YOLO detector."""
        path = self.config.yolo_model_path or self.config.default_model_url
        if path.startswith("http"):
            path = self._download_model(path)
        yolo_model = YOLO(path)
        yolo_model.to(self.config.device)
        return YOLOFaceDetector(yolo_model, conf_threshold=self.config.conf_threshold)

    def _initialize_embedder(self) -> FacenetEmbedder:
        """Initialize Facenet embedder."""
        fn_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.config.device)
        return FacenetEmbedder(fn_model, device=self.config.device, alignment_fn=self.config.alignment_fn)

    def _download_model(self, url: str) -> str:
        """Download and cache model if not already present."""
        base_dir = os.path.join(self.config.cache_dir, "models")
        os.makedirs(base_dir, exist_ok=True)
        model_path = os.path.join(base_dir, os.path.basename(url))

        if not os.path.exists(model_path):
            self.logger.info(f"Downloading model from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            self.logger.info(f"Model saved to {model_path}")
        return model_path

    def detect_faces(self, image: Image.Image) -> Tuple[List[Tuple[int, int, int, int]], Optional[List[np.ndarray]]]:
        if hasattr(self, 'combined_model'):
            results = self.combined_model(image)
            boxes, embeddings = zip(*results) if results else ([], [])
            return list(boxes), list(embeddings)
        else:
            if self.hooks.before_detect:
                image = self.hooks.execute_before_detect(image)
            boxes = self.detector.detect_faces(image)
            if self.hooks.after_detect:
                self.hooks.execute_after_detect(boxes)
            return boxes, None

    def embed_faces_batch(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        if hasattr(self, 'combined_model'):
            _, embeddings = self.detect_faces(image)
            return embeddings
        else:
            if self.hooks.before_embed:
                self.hooks.execute_before_embed(image, boxes)
            embeddings = self.embedder.embed_faces_batch(image, boxes)
            if self.hooks.after_embed:
                self.hooks.execute_after_embed(embeddings)
            return embeddings

    def register_user(self, user_id: str, images: List[Image.Image]) -> str:
        collected_embeddings = []
        for img in images:
            boxes, embeddings = self.detect_faces(img)
            if len(boxes) == 1 and embeddings:
                collected_embeddings.append(embeddings[0])
        if not collected_embeddings:
            return f"[Registration] No valid single-face images found for '{user_id}'."
        if hasattr(self, 'combined_model'):
            if user_id not in self.combined_model.user_embeddings:
                self.combined_model.user_embeddings[user_id] = []
            self.combined_model.user_embeddings[user_id].extend(collected_embeddings)
        else:
            if user_id not in self.user_data:
                self.user_data[user_id] = []
            self.user_data[user_id].extend(collected_embeddings)
            self.data_store.save_user_data(self.user_data)
        return f"[Registration] User '{user_id}' registered with {len(collected_embeddings)} images."

    def identify_user(self, image: Image.Image, threshold=0.6) -> List[Dict[str, Any]]:
        boxes, embeddings = self.detect_faces(image)
        if not boxes:
            return []
        results = []
        for i, emb in enumerate(embeddings):
            best_match = None
            best_sim = 0.0
            user_embeddings = self.combined_model.user_embeddings if hasattr(self, 'combined_model') else self.user_data
            for user_id, emb_list in user_embeddings.items():
                valid_embs = [e for e in emb_list if e.shape == emb.shape]
                if not valid_embs:
                    continue
                sims = cosine_similarity([emb], valid_embs)
                max_sim = np.max(sims)
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_match = user_id
            if best_sim >= threshold:
                results.append({'box': boxes[i], 'user_id': best_match, 'similarity': float(best_sim)})
            else:
                results.append({'box': boxes[i], 'user_id': 'Unknown', 'similarity': float(best_sim)})
        return results

    def export_model(self, export_path: str) -> str:
        if hasattr(self, 'combined_model'):
            self.combined_model.save_model(export_path)
            return f"Combined model saved to {export_path}"
        else:
            state = {
                'yolo_state_dict': self.detector.model.state_dict(),
                'facenet_state_dict': self.embedder.model.state_dict(),
                'user_embeddings': self.user_data,
                'device': self.config.device
            }
            torch.save(state, export_path)
            return f"Model and user data saved to {export_path}"

    @classmethod
    def import_model(cls, import_path: str, config: Config) -> "FacialRecognition":
        combined_model = CombinedFacialRecognitionModel.load_model(import_path, device=config.device)
        return cls(config, combined_model_path=import_path)
