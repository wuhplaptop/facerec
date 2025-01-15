# myfacerec/combined_model.py

import torch
import torch.nn as nn
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import os
from typing import Optional, List, Tuple, Dict
import json

class CombinedFacialRecognitionModel(nn.Module):
    def __init__(
        self, 
        yolo_model_path: str = "yolov8n-face.pt",
        facenet_model: Optional[InceptionResnetV1] = None,
        device: str = 'cpu',
        conf_threshold: float = 0.5
    ):
        super(CombinedFacialRecognitionModel, self).__init__()
        self.device = device
        self.conf_threshold = conf_threshold

        # Initialize YOLO model
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(self.device)

        # Initialize Facenet model
        if facenet_model is None:
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        else:
            self.facenet = facenet_model.eval().to(self.device)

        # User embeddings: Dict[str, List[np.ndarray]]
        self.user_embeddings: Dict[str, List[np.ndarray]] = {}

    def forward(self, image: Image.Image) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Perform face detection and embedding extraction.

        Args:
            image (PIL.Image): Input image.

        Returns:
            List of tuples: Each tuple contains bounding box coordinates and the corresponding embedding.
        """
        # Perform face detection
        detections = self.yolo(image)
        boxes = []
        for result in detections:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= self.conf_threshold and cls == 0:  # Assuming class 0 is 'face'
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        embeddings = []
        for box in boxes:
            face = image.crop(box).resize((160, 160))
            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5
            face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.facenet(face_tensor).cpu().numpy()[0]
            embeddings.append(emb)

        return list(zip(boxes, embeddings))

    def save_model(self, save_path: str):
        """
        Save the combined model and user embeddings to a .pt file.

        Args:
            save_path (str): Path to save the .pt model.
        """
        state = {
            'yolo_state_dict': self.yolo.model.state_dict(),
            'facenet_state_dict': self.facenet.state_dict(),
            'user_embeddings': {user_id: emb_list for user_id, emb_list in self.user_embeddings.items()},
            'config': {
                'yolo_model_path': self.yolo.model_path if hasattr(self.yolo, 'model_path') else "",
                'conf_threshold': self.conf_threshold,
                'device': self.device
            }
        }
        torch.save(state, save_path)
        print(f"Combined model saved to {save_path}.")

    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu') -> "CombinedFacialRecognitionModel":
        """
        Load the combined model and user embeddings from a .pt file.

        Args:
            load_path (str): Path to the .pt model.
            device (str): Device to load the model on.

        Returns:
            CombinedFacialRecognitionModel: Loaded model.
        """
        state = torch.load(load_path, map_location=device)
        config = state['config']
        yolo_model_path = config.get('yolo_model_path', "yolov8n-face.pt")
        conf_threshold = config.get('conf_threshold', 0.5)

        # Initialize model
        model = cls(
            yolo_model_path=yolo_model_path,
            device=device,
            conf_threshold=conf_threshold
        )

        # Load state dictionaries
        model.yolo.model.load_state_dict(state['yolo_state_dict'])
        model.facenet.load_state_dict(state['facenet_state_dict'])
        model.user_embeddings = {user_id: [np.array(e) for e in emb_list] for user_id, emb_list in state['user_embeddings'].items()}

        model.to(device)
        model.eval()
        print(f"Combined model loaded from {load_path} on {device}.")
        return model
