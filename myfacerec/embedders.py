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
    def embed_faces_batch(self, image, boxes: List[Tuple[int, int, int, int]]):
        pass


class FacenetEmbedder(FaceEmbedder):
    def __init__(self, model, device="cpu", alignment_fn=None):
        self.model = model
        self.device = device
        self.alignment_fn = alignment_fn
        self.model.eval()

    def embed_faces_batch(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]):
        if not boxes:
            logger.warning("No face bounding boxes provided.")
            return np.array([])

        face_tensors = []
        for (x1, y1, x2, y2) in boxes:
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
                logger.error(f"Error processing face bounding box ({x1}, {y1}, {x2}, {y2}): {e}")

        if not face_tensors:
            logger.warning("No valid face tensors created. Returning an empty array.")
            return np.array([])

        # Perform embedding generation
        try:
            batch = torch.stack(face_tensors).to(self.device)
            logger.info(f"Processing batch of size {len(face_tensors)} for embeddings.")
            with torch.no_grad():
                embeddings = self.model(batch).cpu().numpy()
            logger.info(f"Generated embeddings for {len(embeddings)} faces.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
