# myfacerec/facial_recognition.py

import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict, Any
from PIL import Image

from .config import Config, logger
from .data_store import JSONUserDataStore, UserDataStore
from .combined_model import CombinedFacialRecognitionModel

class FacialRecognition:
    """
    Main orchestrator for facial recognition that handles face detection,
    embedding generation, user registration, and identification.
    """

    def __init__(
        self,
        config: Config,
        data_store: Optional[UserDataStore] = None,
        model: Optional[CombinedFacialRecognitionModel] = None
    ):
        self.config = config
        self.logger = logger

        # Initialize Data Store
        self.data_store = data_store or JSONUserDataStore(self.config.user_data_path)
        self.user_data = self.data_store.load_user_data()
        self.logger.info("Data store initialized with %d users.", len(self.user_data))

        # Initialize Combined Model
        if model is not None:
            self.model = model
        else:
            self.model = CombinedFacialRecognitionModel(
                yolo_model_path=self.config.yolo_model_path,
                device=self.config.device,
                conf_threshold=self.config.conf_threshold,
                enable_pose_estimation=self.config.enable_pose_estimation  # NEW
            )
        self.model.user_embeddings = self.user_data

    def register_user(self, user_id: str, images: List[Image.Image]) -> str:
        collected_embeddings = []
        for idx, img in enumerate(images):
            try:
                results = self.model(img)  # Each item: { 'box', 'embedding', 'pose' }
                self.logger.info(f"Processing image {idx + 1}: Detected {len(results)} face(s).")

                # Simple logic: Only proceed if exactly 1 face was found
                if len(results) == 1:
                    emb = results[0]['embedding']
                    collected_embeddings.append(emb)
                    self.logger.info(f"Image {idx + 1}: Collected embedding.")
                elif len(results) > 1:
                    self.logger.warning(f"Image {idx + 1}: Multiple faces detected. Skipping.")
                else:
                    self.logger.warning(f"Image {idx + 1}: No faces detected. Skipping.")
            except Exception as e:
                self.logger.error(f"Error processing image {idx + 1}: {e}")

        if not collected_embeddings:
            message = f"No valid face embeddings found for user '{user_id}'."
            self.logger.error(message)
            return message

        # Store embeddings
        if user_id not in self.user_data:
            self.user_data[user_id] = []
        self.user_data[user_id].extend(collected_embeddings)
        self.data_store.save_user_data(self.user_data)
        message = f"User '{user_id}' registered with {len(collected_embeddings)} embedding(s)."
        self.logger.info(message)
        return message

    def identify_user(self, image: Image.Image, threshold: float = 0.6) -> List[Dict[str, Any]]:
        try:
            results = self.model(image)
        except Exception as e:
            self.logger.error(f"Error during face detection and embedding: {e}")
            return []

        # Now we have a list of { 'box', 'embedding', 'pose' }
        embeddings = [res['embedding'] for res in results]
        boxes = [res['box'] for res in results]
        poses = [res['pose'] for res in results]  # might be None if disabled

        identifications = self._identify_embeddings(embeddings, threshold)
        # Combine the identification results with bounding boxes & poses
        combined_results = []
        for i, ident in enumerate(identifications):
            combined_results.append({
                'face_id': i + 1,
                'user_id': ident['user_id'],
                'similarity': ident['similarity'],
                'box': boxes[i],
                'pose': poses[i],
            })

        return combined_results

    def _identify_embeddings(self, embeddings: List[np.ndarray], threshold: float) -> List[Dict[str, Any]]:
        results = []
        for emb in embeddings:
            best_match = None
            best_sim = 0.0

            for user_id, user_embs in self.user_data.items():
                if not user_embs:
                    continue
                sims = cosine_similarity([emb], user_embs)  # shape: (1, N)
                max_sim = sims.max()
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_match = user_id

            if best_sim >= threshold:
                results.append({'user_id': best_match, 'similarity': float(best_sim)})
            else:
                results.append({'user_id': 'Unknown', 'similarity': float(best_sim)})

        return results

    def export_combined_model(self, save_path: str) -> None:
        try:
            self.model.save_model(save_path)
            self.logger.info(f"Combined model exported to {save_path}.")
        except Exception as e:
            self.logger.error(f"Failed to export combined model: {e}")
            raise

    @classmethod
    def import_combined_model(cls, load_path: str, config: Optional[Config] = None) -> "FacialRecognition":
        try:
            model = CombinedFacialRecognitionModel.load_model(load_path)
            user_data = model.user_embeddings

            if config is None:
                config = Config(
                    yolo_model_path=model.yolo.model_path if hasattr(model.yolo, 'model_path') else "myfacerec/models/face.pt",
                    conf_threshold=model.conf_threshold,
                    device=model.device,
                    enable_pose_estimation=model.enable_pose_estimation  # preserve pose
                )

            fr = cls(config=config, model=model)
            fr.user_data = user_data
            fr.data_store.save_user_data(fr.user_data)
            fr.logger.info(f"Combined model imported from {load_path}.")
            return fr
        except Exception as e:
            logger.error(f"Failed to import combined model: {e}")
            raise
