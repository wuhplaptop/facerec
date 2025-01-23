# myfacerec/combined_model.py

import torch
import torch.nn as nn
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import os
from typing import Optional, List, Tuple, Dict

from .pose_estimator import HeadPoseEstimator  # NEW

class CombinedFacialRecognitionModel(nn.Module):
    def __init__(
        self, 
        yolo_model_path: str = "myfacerec/models/face.pt",
        facenet_model: Optional[InceptionResnetV1] = None,
        device: str = 'cpu',
        conf_threshold: float = 0.5,
        enable_pose_estimation: bool = False  # NEW
    ):
        super(CombinedFacialRecognitionModel, self).__init__()
        self.device = device
        self.conf_threshold = conf_threshold
        self.enable_pose_estimation = enable_pose_estimation  # NEW

        # Initialize YOLO model
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(self.device)

        # Initialize Facenet model
        if facenet_model is None:
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        else:
            self.facenet = facenet_model.eval().to(self.device)

        # Pose estimator
        self.pose_estimator = HeadPoseEstimator() if self.enable_pose_estimation else None

        # User embeddings: Dict[user_id, List[np.ndarray]]
        self.user_embeddings: Dict[str, List[np.ndarray]] = {}

    def forward(self, image: Image.Image):
        """
        Perform face detection, embedding, and optional pose estimation.

        Returns:
            A list of dicts: [
              {
                'box': (x1, y1, x2, y2),
                'embedding': np.ndarray,
                'pose': (yaw, pitch, roll) or None
              },
              ...
            ]
        """
        # 1) Perform face detection
        detections = self.yolo(image)
        boxes = []
        for result in detections:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= self.conf_threshold and cls == 0:  # class 0 is 'face'
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Convert image to np array for pose if needed
        if self.enable_pose_estimation:
            image_np = np.array(image)  # e.g. RGB

        outputs = []
        for box in boxes:
            # 2) Crop & embed
            face = image.crop(box).resize((160, 160))
            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5
            face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.facenet(face_tensor).cpu().numpy()[0]

            # 3) If pose estimation is enabled, do it
            pose = None
            if self.enable_pose_estimation and self.pose_estimator:
                # Placeholder for real 2D landmark detection:
                face_landmarks_2d = self._dummy_landmark_detector(box, image_np)
                if face_landmarks_2d is not None:
                    pose = self.pose_estimator.estimate_pose(image_np, face_landmarks_2d)

            outputs.append({
                'box': box,
                'embedding': emb,
                'pose': pose
            })

        return outputs

    def _dummy_landmark_detector(self, box, full_image_np):
        """
        Placeholder for a real 2D landmark detection method.
        Returns np.ndarray of shape (6,2) or None if not detected.

        You must replace this with a real landmark detector. 
        """
        # This is just a dummy example returning random points near the face center:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w < 10 or h < 10:
            return None

        # Return 6 random points for demonstration
        points = []
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        for i in range(6):
            px = center_x + (np.random.rand() - 0.5) * 0.3 * w
            py = center_y + (np.random.rand() - 0.5) * 0.3 * h
            points.append([px, py])
        return np.array(points, dtype=np.float32)

    def save_model(self, save_path: str):
        """
        Save the combined model and user embeddings to a .pt file.
        """
        state = {
            'yolo_state_dict': self.yolo.model.state_dict(),
            'facenet_state_dict': self.facenet.state_dict(),
            'user_embeddings': {uid: emb for uid, emb in self.user_embeddings.items()},
            'config': {
                'yolo_model_path': self.yolo.model_path if hasattr(self.yolo, 'model_path') else "",
                'conf_threshold': self.conf_threshold,
                'device': self.device,
                'enable_pose_estimation': self.enable_pose_estimation,
            }
        }
        torch.save(state, save_path)
        print(f"Combined model saved to {save_path}.")

    @classmethod
    def load_model(cls, load_path: str, device: str = 'cpu') -> "CombinedFacialRecognitionModel":
        """
        Load the combined model and user embeddings from a .pt file.
        """
        state = torch.load(load_path, map_location=device)
        config = state['config']
        yolo_model_path = config.get('yolo_model_path', "myfacerec/models/face.pt")
        conf_threshold = config.get('conf_threshold', 0.5)
        enable_pose_estimation = config.get('enable_pose_estimation', False)

        # Initialize the model
        model = cls(
            yolo_model_path=yolo_model_path,
            device=device,
            conf_threshold=conf_threshold,
            enable_pose_estimation=enable_pose_estimation
        )

        # Load state dictionaries
        model.yolo.model.load_state_dict(state['yolo_state_dict'])
        model.facenet.load_state_dict(state['facenet_state_dict'])
        model.user_embeddings = {
            uid: [np.array(e) for e in emb_list]
            for uid, emb_list in state['user_embeddings'].items()
        }

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

