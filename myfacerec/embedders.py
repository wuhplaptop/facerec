# myfacerec/embedders.py

import abc
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceEmbedder(abc.ABC):
    """Abstract base class for face embedding."""
    
    @abc.abstractmethod
    def embed_faces_batch(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Generate embeddings for a batch of face bounding boxes in an image.
        
        Args:
            image (PIL.Image.Image): The input image.
            boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x1, y1, x2, y2).
        
        Returns:
            np.ndarray: Array of embeddings with shape (num_faces, embedding_dim).
        """
        pass

class FacenetEmbedder(FaceEmbedder):
    def __init__(self, model, device="cpu", alignment_fn=None):
        """
        Initialize the Facenet embedder.
        
        Args:
            model (torch.nn.Module): The Facenet model.
            device (str): Device to run the model on ('cpu' or 'cuda').
            alignment_fn (callable, optional): Function to align faces.
        """
        self.model = model
        self.device = device
        self.alignment_fn = alignment_fn
        self.model.eval()
        self.model.to(self.device)
        logger.info("FacenetEmbedder initialized on device: %s", self.device)

    def embed_faces_batch(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Generate embeddings for a batch of face bounding boxes in an image.
        
        Args:
            image (PIL.Image.Image): The input image.
            boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x1, y1, x2, y2).
        
        Returns:
            np.ndarray: Array of embeddings with shape (num_faces, embedding_dim).
        """
        if not boxes:
            logger.warning("No face bounding boxes provided.")
            return np.array([])

        face_tensors = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            try:
                # Crop and resize the face
                face = image.crop((x1, y1, x2, y2)).resize((160, 160))
                if self.alignment_fn:
                    face = self.alignment_fn(face)

                # Normalize the face image
                face_np = np.array(face).astype(np.float32) / 255.0
                face_np = (face_np - 0.5) / 0.5

                # Check shape validity
                if face_np.shape != (160, 160, 3):
                    logger.error(f"Invalid face shape: {face_np.shape}. Expected (160, 160, 3). Skipping.")
                    continue

                face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).float()
                face_tensors.append(face_tensor)
            except Exception as e:
                logger.error(f"Error processing face bounding box {idx + 1} ({x1}, {y1}, {x2}, {y2}): {e}")

        if not face_tensors:
            logger.warning("No valid face tensors created. Returning an empty array.")
            return np.array([])

        # Perform embedding generation
        try:
            batch = torch.stack(face_tensors).to(self.device)  # Shape: (num_faces, 3, 160, 160)
            logger.info(f"Processing batch of size {len(face_tensors)} for embeddings.")
            with torch.no_grad():
                embeddings = self.model(batch).cpu().numpy()  # Shape: (num_faces, embedding_dim)
            logger.info(f"Generated embeddings for {len(embeddings)} face(s).")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
