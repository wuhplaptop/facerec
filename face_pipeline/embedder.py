# face_pipeline/embedder.py

import cv2
import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

logger = logging.getLogger(__name__)

class FaceNetEmbedder:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb).convert('RGB')
            tens = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(tens)[0].cpu().numpy()
            logger.debug(f"Generated embedding: {embedding[:5]}...")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return None
