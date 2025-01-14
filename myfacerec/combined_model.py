# myfacerec/combined_model.py

import torch
import torch.nn as nn
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import json
import os

class CombinedFacialRecognitionModel(nn.Module):
    def __init__(self, yolo_model_path: str, device: str = 'cpu'):
        super(CombinedFacialRecognitionModel, self).__init__()
        # Initialize YOLO model
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(device)

        # Initialize Facenet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # User embeddings: Dict[str, List[np.ndarray]]
        self.user_embeddings = {}
        self.device = device

    def forward(self, image):
        """
        Perform face detection and embedding extraction.

        Args:
            image (PIL.Image or torch.Tensor): Input image.

        Returns:
            List of tuples: Each tuple contains bounding box coordinates and the corresponding embedding.
        """
        # Ensure image is a PIL Image
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype('uint8'))

        # Detect faces using YOLO
        detections = self.yolo(image)
        boxes = []
        for result in detections:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= 0.5 and cls == 0:  # Assuming class 0 is 'face'
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
        # Prepare state dictionary
        state = {
            'yolo_state_dict': self.yolo.model.state_dict(),
            'facenet_state_dict': self.facenet.state_dict(),
            'user_embeddings': self.user_embeddings,
            'device': self.device
        }
        torch.save(state, save_path)

    @classmethod
    def load_model(cls, load_path: str):
        """
        Load the combined model and user embeddings from a .pt file.

        Args:
            load_path (str): Path to the .pt model.

        Returns:
            CombinedFacialRecognitionModel: Loaded model.
        """
        state = torch.load(load_path, map_location='cpu')
        device = state.get('device', 'cpu')
        model = cls(yolo_model_path='', device=device)  # Temporary path; will load state_dict next

        # Load YOLO state_dict
        model.yolo.model.load_state_dict(state['yolo_state_dict'])

        # Load Facenet state_dict
        model.facenet.load_state_dict(state['facenet_state_dict'])

        # Load user embeddings
        model.user_embeddings = state['user_embeddings']

        model.to(device)
        model.eval()
        return model
