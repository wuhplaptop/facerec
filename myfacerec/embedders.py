# embedders.py

import abc
import numpy as np
import torch
from PIL import Image

class FaceEmbedder(abc.ABC):
    """Abstract base class for face embedding."""
    @abc.abstractmethod
    def embed_faces_batch(self, image, boxes):
        pass


class FacenetEmbedder(FaceEmbedder):
    def __init__(self, model, device="cpu", alignment_fn=None):
        self.model = model
        self.device = device
        self.alignment_fn = alignment_fn
        self.model.eval()

    def embed_faces_batch(self, image, boxes):
        if not boxes:
            return np.array([])

        face_tensors = []
        for (x1, y1, x2, y2) in boxes:
            face = image.crop((x1, y1, x2, y2)).resize((160, 160))

            if self.alignment_fn:
                face = self.alignment_fn(face)

            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5
            face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).float()
            face_tensors.append(face_tensor)

        if not face_tensors:
            return np.array([])

        batch = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()
        return embeddings
