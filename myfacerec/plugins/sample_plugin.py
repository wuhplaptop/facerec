# myfacerec/plugins/sample_plugin.py

from ..embedders import FaceEmbedder
from PIL import Image
import numpy as np
import torch

class SampleEmbedder(FaceEmbedder):
    def __init__(self):
        # Initialize your custom embedder here
        pass

    def embed_faces_batch(self, image: Image.Image, boxes: List[Tuple[int, int, int, int]]):
        # Implement custom embedding logic
        # For demonstration, return a dummy embedding
        if not boxes:
            return np.array([])
        return np.array([np.random.rand(512).astype(np.float32) for _ in boxes])
