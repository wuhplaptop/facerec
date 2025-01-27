# face_pipeline/database.py

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class FaceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded database from {self.db_path}")
        except Exception as e:
            logger.error(f"Database load failed: {str(e)}")
            self.embeddings = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Saved database to {self.db_path}")
        except Exception as e:
            logger.error(f"Database save failed: {str(e)}")
            raise RuntimeError(f"Database save failed: {str(e)}") from e

    def add_embedding(self, label: str, embedding: np.ndarray):
        try:
            if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
                raise ValueError("Invalid embedding format")
            if label not in self.embeddings:
                self.embeddings[label] = []
            self.embeddings[label].append(embedding)
            logger.debug(f"Added embedding for {label}")
        except Exception as e:
            logger.error(f"Add embedding failed: {str(e)}")
            raise

    def remove_label(self, label: str):
        try:
            if label in self.embeddings:
                del self.embeddings[label]
                logger.info(f"Removed {label}")
            else:
                logger.warning(f"Label {label} not found")
        except Exception as e:
            logger.error(f"Remove label failed: {str(e)}")
            raise

    def list_labels(self) -> List[str]:
        return list(self.embeddings.keys())

    def get_embeddings_by_label(self, label: str) -> Optional[List[np.ndarray]]:
        return self.embeddings.get(label)

    def search_by_image(self, query_embedding: np.ndarray, threshold: float = 0.7) -> List[Tuple[str, float]]:
        results = []
        for label, embeddings in self.embeddings.items():
            for db_emb in embeddings:
                similarity = FacePipeline.cosine_similarity(query_embedding, db_emb)
                if similarity >= threshold:
                    results.append((label, similarity))
        return sorted(results, key=lambda x: x[1], reverse=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
