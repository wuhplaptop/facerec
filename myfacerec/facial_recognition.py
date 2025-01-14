# myfacerec/facial_recognition.py

import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Dict, Any

from PIL import Image

from .config import Config, logger
from .detectors import YOLOFaceDetector, FaceDetector
from .embedders import FacenetEmbedder, FaceEmbedder
from .data_store import JSONUserDataStore, UserDataStore

import torch

class FacialRecognition:
    """
    Main orchestrator for facial recognition that handles face detection,
    embedding generation, user registration, and identification.
    """

    def __init__(
        self,
        config: Config,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        data_store: Optional[UserDataStore] = None
    ):
        """
        Initialize the FacialRecognition class with detector, embedder, and data store.

        Args:
            config (Config): Configuration object.
            detector (FaceDetector, optional): Custom face detector.
            embedder (FaceEmbedder, optional): Custom face embedder.
            data_store (UserDataStore, optional): Custom data store.
        """
        self.config = config
        self.logger = logger

        # Initialize Detector
        if detector:
            self.detector = detector
            self.logger.info("Custom FaceDetector provided.")
        else:
            self.detector = self._initialize_detector()
            self.logger.info("YOLOFaceDetector initialized.")

        # Initialize Embedder
        if embedder:
            self.embedder = embedder
            self.logger.info("Custom FaceEmbedder provided.")
        else:
            self.embedder = self._initialize_embedder()
            self.logger.info("FacenetEmbedder initialized.")

        # Initialize Data Store
        if data_store:
            self.data_store = data_store
            self.user_data = self.data_store.load_user_data()
            self.logger.info("Custom UserDataStore provided.")
        else:
            self.data_store = JSONUserDataStore(self.config.user_data_path)
            self.user_data = self.data_store.load_user_data()
            self.logger.info("JSONUserDataStore initialized with %d users.", len(self.user_data))

    def _initialize_detector(self) -> YOLOFaceDetector:
        """
        Initialize the YOLO face detector.

        Returns:
            YOLOFaceDetector: Initialized YOLOFaceDetector instance.
        """
        if self.config.yolo_model_path.startswith("http"):
            model_path = self._download_model(self.config.yolo_model_path)
        else:
            model_path = self.config.yolo_model_path

        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        yolo_model.to(self.config.device)
        self.logger.info("YOLO model loaded from %s on device %s.", model_path, self.config.device)
        return YOLOFaceDetector(yolo_model, conf_threshold=self.config.conf_threshold)

    def _initialize_embedder(self) -> FacenetEmbedder:
        """
        Initialize the Facenet embedder.

        Returns:
            FacenetEmbedder: Initialized FacenetEmbedder instance.
        """
        # Initialize Facenet model
        facenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_resnet_v1', pretrained='vggface2')
        facenet_model.eval()
        facenet_model.to(self.config.device)
        self.logger.info("Facenet model loaded on device %s.", self.config.device)

        return FacenetEmbedder(facenet_model, device=self.config.device, alignment_fn=self.config.alignment_fn)

    def _download_model(self, url: str) -> str:
        """
        Download a model from a URL and cache it locally.

        Args:
            url (str): URL to download the model from.

        Returns:
            str: Local path to the downloaded model.
        """
        import requests

        base_dir = os.path.join(self.config.cache_dir, "models")
        os.makedirs(base_dir, exist_ok=True)
        model_filename = os.path.basename(url)
        model_path = os.path.join(base_dir, model_filename)

        if not os.path.exists(model_path):
            self.logger.info(f"Downloading model from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            self.logger.info(f"Model downloaded and saved to {model_path}")
        else:
            self.logger.info(f"Model already exists at {model_path}")

        return model_path

    def detect_faces(self, image: Image.Image) -> Tuple[List[Tuple[int, int, int, int]], Optional[List[np.ndarray]]]:
        """
        Detect faces in an image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], Optional[List[np.ndarray]]]:
                - List of bounding boxes (x1, y1, x2, y2).
                - List of embeddings if using a combined model, else None.
        """
        boxes = self.detector.detect_faces(image)
        self.logger.info(f"Detected {len(boxes)} face(s).")
        return boxes, None  # Since embeddings are handled separately

    def embed_faces_batch(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Generate embeddings for a batch of face bounding boxes.

        Args:
            image (PIL.Image.Image): The input image.
            boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x1, y1, x2, y2).

        Returns:
            np.ndarray: Array of embeddings with shape (num_faces, embedding_dim).
        """
        embeddings = self.embedder.embed_faces_batch(image, boxes)
        self.logger.info(f"Generated {embeddings.shape[0]} embedding(s).")
        return embeddings

    def register_user(self, user_id: str, images: List[Image.Image]) -> str:
        """
        Register a user with their face embeddings.

        Args:
            user_id (str): Unique identifier for the user.
            images (List[PIL.Image.Image]): List of images containing the user's face.

        Returns:
            str: Registration status message.
        """
        collected_embeddings = []
        for idx, img in enumerate(images):
            boxes, _ = self.detect_faces(img)
            self.logger.info(f"Processing image {idx + 1}: Detected {len(boxes)} face(s).")

            if len(boxes) == 1:
                embeddings = self.embed_faces_batch(img, boxes)
                if embeddings.size == 0:
                    self.logger.warning(f"Embedding generation failed for image {idx + 1}. Skipping.")
                    continue
                self.logger.info(f"Image {idx + 1}: Collected embedding of shape {embeddings[0].shape}.")
                collected_embeddings.append(embeddings[0])
            elif len(boxes) > 1:
                self.logger.warning(f"Image {idx + 1}: Multiple faces detected. Skipping.")
            else:
                self.logger.warning(f"Image {idx + 1}: No faces detected. Skipping.")

        if not collected_embeddings:
            message = f"No valid face embeddings found for user '{user_id}'."
            self.logger.error(message)
            return message

        # Store embeddings
        if user_id not in self.user_data:
            self.user_data[user_id] = []
        self.user_data[user_id].extend(collected_embeddings)
        self.data_store.save_user_data(self.user_data)
        message = f"User '{user_id}' registered with {len(collected_embeddings)} valid face embedding(s)."
        self.logger.info(message)
        return message

    def identify_user(self, embeddings: np.ndarray, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Identify users based on the provided embeddings.

        Args:
            embeddings (np.ndarray): Array of face embeddings to identify. Shape: (num_faces, embedding_dim)
            threshold (float): Similarity threshold for recognition.

        Returns:
            List[Dict[str, Any]]: List of identification results with bounding boxes, user IDs, and similarity scores.
        """
        if embeddings.ndim != 2:
            self.logger.error(f"Embeddings should be a 2D array, got {embeddings.ndim}D array instead.")
            raise ValueError("Embeddings should be a 2D array.")

        results = []
        for idx, emb in enumerate(embeddings):
            if emb.ndim != 1:
                self.logger.error(f"Embedding at index {idx} has incorrect dimensions: {emb.shape}")
                results.append({'user_id': 'Unknown', 'similarity': 0.0})
                continue

            best_match = None
            best_sim = 0.0

            for user_id, emb_list in self.user_data.items():
                if not emb_list:
                    continue
                try:
                    emb_array = np.stack(emb_list)  # Shape: (num_user_embeddings, embedding_dim)
                except ValueError as e:
                    self.logger.error(f"Error stacking embeddings for user {user_id}: {e}")
                    continue
                if emb_array.shape[1] != emb.shape[0]:
                    self.logger.error(f"Embedding dimension mismatch for user '{user_id}': {emb_array.shape[1]} vs {emb.shape[0]}")
                    continue
                sims = cosine_similarity([emb], emb_array)  # Shape: (1, num_user_embeddings)
                max_sim = sims.max()
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_match = user_id

            if best_sim >= threshold:
                results.append({'user_id': best_match, 'similarity': float(best_sim)})
                self.logger.info(f"Embedding {idx + 1}: Matched with user '{best_match}' (Similarity: {best_sim:.2f}).")
            else:
                results.append({'user_id': 'Unknown', 'similarity': float(best_sim)})
                self.logger.info(f"Embedding {idx + 1}: No match found (Highest Similarity: {best_sim:.2f}).")

        return results

    def export_model(self, save_path: str) -> None:
        """
        Export the current state of the facial recognition model.

        Args:
            save_path (str): Path to save the exported model.
        """
        try:
            state = {
                'yolo_state_dict': self.detector.model.state_dict(),
                'facenet_state_dict': self.embedder.model.state_dict(),
                'user_embeddings': {user_id: emb_list for user_id, emb_list in self.user_data.items()},
                'device': self.config.device
            }
            torch.save(state, save_path)
            self.logger.info(f"Model and user data exported to {save_path}.")
        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise

    @classmethod
    def import_model(cls, import_path: str, config: Config) -> "FacialRecognition":
        """
        Import a facial recognition model from a saved state.

        Args:
            import_path (str): Path to the exported model file.
            config (Config): Configuration object.

        Returns:
            FacialRecognition: Initialized FacialRecognition instance with imported state.
        """
        try:
            state = torch.load(import_path, map_location=config.device)
            detector = YOLOFaceDetector(
                torch.hub.load('ultralytics/yolov5', 'custom', path=config.yolo_model_path, force_reload=True),
                conf_threshold=config.conf_threshold
            )
            embedder = FacenetEmbedder(
                torch.hub.load('pytorch/vision:v0.10.0', 'inception_resnet_v1', pretrained='vggface2'),
                device=config.device,
                alignment_fn=config.alignment_fn
            )
            data_store = JSONUserDataStore(config.user_data_path)
            fr = cls(config, detector=detector, embedder=embedder, data_store=data_store)

            # Load state dictionaries
            fr.detector.model.load_state_dict(state['yolo_state_dict'])
            fr.embedder.model.load_state_dict(state['facenet_state_dict'])
            fr.user_data = {user_id: [np.array(e) for e in emb_list] for user_id, emb_list in state['user_embeddings'].items()}
            fr.logger.info(f"Model imported from {import_path}.")
            return fr
        except KeyError as e:
            fr.logger.error(f"Missing key in the imported model state: {e}")
            raise
        except Exception as e:
            fr.logger.error(f"Failed to import model: {e}")
            raise
