# embedders.py

import abc
import numpy as np
import torch
from PIL import Image

class FaceEmbedder(abc.ABC):
    """
    Abstract base class for any face embedding approach.
    """
    @abc.abstractmethod
    def embed_faces_batch(self, image, boxes):
        """
        Given a PIL image and bounding boxes, return embeddings as np.ndarray.
        """
        pass


class FacenetEmbedder(FaceEmbedder):
    def __init__(self, model, device="cpu", alignment_fn=None):
        """
        Args:
            model: A loaded Facenet model (facenet_pytorch).
            device (str): "cpu" or "cuda"
            alignment_fn (callable): optional function that processes
                                     each face before embedding.
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.alignment_fn = alignment_fn

    def embed_faces_batch(self, image, boxes):
        if not boxes:
            return np.array([])

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Embedding {len(boxes)} faces...")

        face_tensors = []
        for (x1,y1,x2,y2) in boxes:
            face = image.crop((x1,y1,x2,y2)).resize((160,160))
            if self.alignment_fn:
                face = self.alignment_fn(face)  # call user-supplied alignment

            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5  # [-1,1]
            face_tensor = torch.from_numpy(face_np).permute(2,0,1).float()
            face_tensors.append(face_tensor)

        if not face_tensors:
            return np.array([])

        batch = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()
        return embeddings
