# myfacerec/facial_recognition.py

import os
import requests
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple

from PIL import Image

from .config import Config, logger
from .detectors import YOLOFaceDetector, FaceDetector
from .embedders import FacenetEmbedder, FaceEmbedder
from .data_store import JSONUserDataStore, UserDataStore
from .hooks import Hooks
from .plugins.base import PluginManager
from .combined_model import CombinedFacialRecognitionModel  # Import the combined model

import torch
from ultralytics import YOLO  # Ensure YOLO is imported
from facenet_pytorch import InceptionResnetV1  # Ensure InceptionResnetV1 is imported


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
        detector: FaceDetector = None,
        embedder: FaceEmbedder = None,
        data_store: UserDataStore = None,
        combined_model_path: str = None  # New parameter for combined model
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hooks = Hooks()

        # Initialize Plugin Manager
        self.plugin_manager = PluginManager()

        # Initialize Combined Model if provided
        if combined_model_path:
            self.combined_model = CombinedFacialRecognitionModel(
                yolo_model_path=config.yolo_model_path or "yolov8n.pt",
                device=config.device
            )
            self.combined_model = CombinedFacialRecognitionModel.load_model(combined_model_path)
            self.logger.info("Initialized with combined YOLO and Facenet model.")
        else:
            # If plugins are specified, load them
            if config.detector_plugin:
                detector = self.plugin_manager.load_detector(config.detector_plugin)
            if config.embedder_plugin:
                embedder = self.plugin_manager.load_embedder(config.embedder_plugin)

            # If no detector, set up YOLO by default
            if detector is None:
                if self.config.yolo_model_path:
                    path = self.config.yolo_model_path
                    if path.startswith("http"):
                        path = self._download_custom_model(path)
                else:
                    path = self._download_model_if_needed(self.config.default_model_url)

                yolo_model = YOLO(path)
                yolo_model.to(config.device)
                detector = YOLOFaceDetector(yolo_model, conf_threshold=config.conf_threshold)
            self.detector = detector

            # If no embedder, use Facenet
            if embedder is None:
                fn_model = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)
                embedder = FacenetEmbedder(
                    fn_model,
                    device=config.device,
                    alignment_fn=config.alignment_fn
                )
            self.embedder = embedder

            # If no data store, use JSON
            if data_store is None:
                data_store = JSONUserDataStore(config.user_data_path)
            self.data_store = data_store

            # Load user data
            self.user_data = self.data_store.load_user_data()
            self.logger.info("Initialized with %d users in data store.", len(self.user_data))

    def _download_model_if_needed(self, url):
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
        if hasattr(self, 'combined_model'):
            # Use combined model for detection and embedding
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

    def embed_faces_batch(self, image, boxes):
        if hasattr(self, 'combined_model'):
            # Embeddings are already obtained from the combined model
            # This method can return the embeddings alongside boxes
            _, embeddings = self.detect_faces(image)
            return embeddings
        else:
            if self.hooks.before_embed:
                self.hooks.execute_before_embed(image, boxes)
            embeddings = self.embedder.embed_faces_batch(image, boxes)
            if self.hooks.after_embed:
                self.hooks.execute_after_embed(embeddings)
            return embeddings

    def register_user(self, user_id: str, images: List[Image.Image]):
        collected_embeddings = []
        for img in images:
            boxes, embeddings = self.detect_faces(img)
            if len(boxes) == 1 and embeddings:
                collected_embeddings.append(embeddings[0])
        if not collected_embeddings:
            return f"[Registration] No valid single-face images found for '{user_id}'."
        if hasattr(self, 'combined_model'):
            # Update the combined model's user embeddings
            if user_id not in self.combined_model.user_embeddings:
                self.combined_model.user_embeddings[user_id] = []
            self.combined_model.user_embeddings[user_id].extend(collected_embeddings)
        else:
            # Update the separate data store
            if user_id not in self.user_data:
                self.user_data[user_id] = []
            self.user_data[user_id].extend(collected_embeddings)
            self.data_store.save_user_data(self.user_data)
        return f"[Registration] User '{user_id}' registered with {len(collected_embeddings)} images."

    def identify_user(self, image: Image.Image, threshold=0.6):
        boxes, embeddings = self.detect_faces(image)
        if not boxes:
            return []
        results = []
        for i, emb in enumerate(embeddings):
            best_match = None
            best_sim = 0.0
            if hasattr(self, 'combined_model'):
                user_embeddings = self.combined_model.user_embeddings
            else:
                user_embeddings = self.user_data
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
        if hasattr(self, 'combined_model'):
            return list(self.combined_model.user_embeddings.keys())
        else:
            return list(self.user_data.keys())

    def delete_user(self, user_id: str):
        if hasattr(self, 'combined_model'):
            if user_id in self.combined_model.user_embeddings:
                del self.combined_model.user_embeddings[user_id]
                self.combined_model.save_model(self.config.user_data_path)  # Reuse save_path for combined model
                return True
            return False
        else:
            if user_id in self.user_data:
                del self.user_data[user_id]
                self.data_store.save_user_data(self.user_data)
                return True
            return False

    def update_user_embeddings(self, user_id: str, new_embeddings: List[np.ndarray]):
        if hasattr(self, 'combined_model'):
            if user_id not in self.combined_model.user_embeddings:
                return False
            self.combined_model.user_embeddings[user_id].extend(new_embeddings)
            self.combined_model.save_model(self.config.user_data_path)
            return True
        else:
            if user_id not in self.user_data:
                return False
            self.user_data[user_id].extend(new_embeddings)
            self.data_store.save_user_data(self.user_data)
            return True

    def export_model(self, export_path: str):
        """
        Export the current state (models and user embeddings) to a .pt file.

        Args:
            export_path (str): Path to save the exported .pt model.
        """
        if hasattr(self, 'combined_model'):
            self.combined_model.save_model(export_path)
            return f"[Export] Combined model saved to {export_path}"
        else:
            # Export separate models and user data
            # Create a dictionary to save all necessary components
            state = {
                'yolo_state_dict': self.detector.model.state_dict(),
                'facenet_state_dict': self.embedder.model.state_dict(),
                'user_embeddings': self.user_data,
                'device': self.config.device
            }
            torch.save(state, export_path)
            return f"[Export] Model and user data saved to {export_path}"

    @classmethod
    def import_model(cls, import_path: str, config: Config):
        """
        Import a combined model from a .pt file.

        Args:
            import_path (str): Path to the .pt model.
            config (Config): Configuration object.

        Returns:
            FacialRecognition: Initialized FacialRecognition instance with the imported model.
        """
        combined_model = CombinedFacialRecognitionModel.load_model(import_path)
        fr = cls(config, combined_model_path=import_path)
        return fr
