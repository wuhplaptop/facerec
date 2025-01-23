# myfacerec/combined_model.py

import torch
import torch.nn as nn
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import os
from typing import Optional, List, Tuple, Dict

from .pose_estimator import HeadPoseEstimator  # If you have pose code
# from .pose_estimator import HeadPoseEstimator

class CombinedFacialRecognitionModel(nn.Module):
    def __init__(
        self, 
        yolo_model_path: str = "myfacerec/models/face.pt",
        facenet_model: Optional[InceptionResnetV1] = None,
        device: str = 'cpu',
        conf_threshold: float = 0.5,
        enable_pose_estimation: bool = False
    ):
        super(CombinedFacialRecognitionModel, self).__init__()
        self.device = device
        self.conf_threshold = conf_threshold
        self.enable_pose_estimation = enable_pose_estimation

        # Initialize YOLO model
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(self.device)

        # Initialize Facenet model
        if facenet_model is None:
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        else:
            self.facenet = facenet_model.eval().to(self.device)

        # (Optional) Pose Estimator
        self.pose_estimator = HeadPoseEstimator() if self.enable_pose_estimation else None

        # User embeddings: Dict[user_id, List[np.ndarray]]
        self.user_embeddings: Dict[str, List[np.ndarray]] = {}

    def forward(self, image: Image.Image):
        """
        Perform face detection, embedding, and optional pose estimation.
        Returns: list of dicts with keys: 'box', 'embedding', 'pose'
        """
        detections = self.yolo(image)
        boxes = []
        for result in detections:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= self.conf_threshold and cls == 0:  # class 0 = 'face'
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        outputs = []
        image_np = None
        if self.enable_pose_estimation:
            image_np = np.array(image)

        for box in boxes:
            # Crop & prepare for FaceNet
            face = image.crop(box).resize((160, 160))
            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5
            face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.facenet(face_tensor).cpu().numpy()[0]

            # Optional pose
            pose = None
            if self.enable_pose_estimation and self.pose_estimator is not None:
                # dummy or real landmark detection here
                # pose = self.pose_estimator.estimate_pose(...)
                pass

            outputs.append({
                'box': box,
                'embedding': emb,
                'pose': pose
            })

        return outputs

    def save_model(self, save_path: str):
        """
        Save the combined model + user embeddings to a .pt file.
        Note: We call self.facenet.state_dict(), not self.facenet.model.state_dict().
        """
        state = {
            'yolo_state_dict': self.yolo.model.state_dict(),
            'facenet_state_dict': self.facenet.state_dict(),   # FIXED
            'user_embeddings': self.user_embeddings,
            'config': {
                'yolo_model_path': getattr(self.yolo, 'model_path', ""),
                'conf_threshold': self.conf_threshold,
                'device': self.device,
                'enable_pose_estimation': self.enable_pose_estimation
            }
        }
        torch.save(state, save_path)
        print(f"Combined model saved to {save_path}.")

    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu') -> "CombinedFacialRecognitionModel":
        """
        Load from .pt file. We'll create a new YOLO + InceptionResnetV1
        and then load the state dictionaries. 
        """
        state = torch.load(load_path, map_location=device)
        config = state['config']
        yolo_model_path = config.get('yolo_model_path', "myfacerec/models/face.pt")
        conf_threshold = config.get('conf_threshold', 0.5)
        enable_pose_estimation = config.get('enable_pose_estimation', False)

        # Initialize YOLO + FaceNet
        model = cls(
            yolo_model_path=yolo_model_path,
            device=device,
            conf_threshold=conf_threshold,
            enable_pose_estimation=enable_pose_estimation
        )

        # Load YOLO weights
        model.yolo.model.load_state_dict(state['yolo_state_dict'])

        # Load FaceNet weights (NO .model, just self.facenet)
        model.facenet.load_state_dict(state['facenet_state_dict'])

        # User embeddings
        model.user_embeddings = state['user_embeddings']

        model.to(device)
        model.eval()
        print(f"Combined model loaded from {load_path} on {device}. Pose estimation = {enable_pose_estimation}.")
        return model


    def detect_and_embed(self, image: Image.Image):
        """
        For compatibility: returns a list of (box, embedding) but not the pose.
        """
        results = self.forward(image)
        return [(res['box'], res['embedding']) for res in results]

