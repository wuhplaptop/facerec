"""
facial_recognition.py

A universal facial recognition library that:
 - Uses a YOLO model for face detection
 - Uses Facenet (InceptionResnetV1) for embeddings
 - Stores embeddings in a JSON file
 - Can download a default YOLO model from config.yaml if no custom path is specified
 - Allows registration and identification of faces
"""

import os
import json
import requests
import yaml
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

class FacialRecognition:
    def __init__(self, 
                 yolo_model_path: str = None, 
                 user_data_path: str = 'user_faces.json',
                 conf_threshold: float = None,
                 use_gpu: bool = True):
        """
        Initialize the facial recognition system.

        Args:
            yolo_model_path (str, optional): Path to YOLO .pt. If None, will use default from config.yaml.
            user_data_path (str): JSON file path for user embeddings.
            conf_threshold (float, optional): Confidence threshold for YOLO. If None, read from config.yaml.
            use_gpu (bool): If True (and CUDA available), use GPU for YOLO + Facenet.
        """
        # Load configuration from config.yaml
        config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        self.default_model_url = config.get("default_model_url", "")
        default_conf_threshold = config.get("default_conf_threshold", 0.5)

        # If user did not supply a model path, download the default model if needed
        if yolo_model_path is None:
            yolo_model_path = self._download_default_model_if_needed(self.default_model_url)

        # If user did not supply a confidence threshold, use from config
        if conf_threshold is None:
            conf_threshold = default_conf_threshold

        self.yolo_model_path = yolo_model_path
        self.user_data_path = user_data_path
        self.conf_threshold = conf_threshold

        # Decide device
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        # Load YOLO + Facenet
        self.yolo_model = self._load_yolo_model()
        self.facenet_model = self._load_facenet_model()

        # Load user embeddings from JSON
        self.user_data = self._load_user_data()

    def _download_default_model_if_needed(self, model_url: str):
        """
        If user doesn't provide a YOLO .pt path, download the default from 'model_url'
        and store it in ~/.myfacerec/face.pt.
        Returns:
            str: The local path to the default model.
        """
        if not model_url:
            raise ValueError("No default model URL found in config.yaml.")

        base_dir = os.path.expanduser("~/.myfacerec")
        os.makedirs(base_dir, exist_ok=True)
        default_model_path = os.path.join(base_dir, "face.pt")

        # Download if not present
        if not os.path.exists(default_model_path):
            print(f"Downloading default YOLO model from: {model_url}")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            with open(default_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Default YOLO model saved to {default_model_path}.")

        return default_model_path

    def _load_yolo_model(self):
        """
        Loads YOLO from self.yolo_model_path, moves to self.device.
        """
        try:
            model = YOLO(self.yolo_model_path)
            model.to(self.device)
            return model
        except Exception as e:
            raise ValueError(f"Error loading YOLO model from {self.yolo_model_path}: {e}")

    def _load_facenet_model(self):
        """
        Loads InceptionResnetV1 (Facenet) on self.device.
        """
        try:
            model = InceptionResnetV1(pretrained='vggface2').eval()
            model = model.to(self.device)
            return model
        except Exception as e:
            raise ValueError(f"Error loading Facenet model: {e}")

    def _load_user_data(self):
        """
        Load user embeddings from JSON.
        """
        if not os.path.exists(self.user_data_path):
            with open(self.user_data_path, 'w') as f:
                json.dump({}, f)

        with open(self.user_data_path, 'r') as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        for user_id in data:
            data[user_id] = [np.array(e) for e in data[user_id]]
        return data

    def _save_user_data(self):
        """
        Save user_data to JSON.
        """
        serializable_data = {
            user_id: [emb.tolist() for emb in emb_list]
            for user_id, emb_list in self.user_data.items()
        }
        with open(self.user_data_path, 'w') as f:
            json.dump(serializable_data, f)

    def detect_faces(self, pil_image):
        """
        Detect faces in a PIL image using YOLO.

        Returns list of (x1, y1, x2, y2).
        """
        results = self.yolo_model(pil_image)
        boxes = []
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= self.conf_threshold and cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def encode_faces_batch(self, pil_image, boxes):
        """
        Batch-encode faces for speed.
        Returns an np.ndarray of shape (num_faces, 512).
        """
        if not boxes:
            return np.array([])

        face_tensors = []
        for (x1, y1, x2, y2) in boxes:
            face = pil_image.crop((x1, y1, x2, y2))
            face = face.resize((160, 160))
            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5
            face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).float()
            face_tensors.append(face_tensor)

        batch = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.facenet_model(batch).cpu().numpy()
        return embeddings

    def register_user(self, user_id: str, pil_images: list):
        """
        Register user by extracting face embeddings (single face per image).

        Returns a status message.
        """
        collected_embeddings = []
        for img in pil_images:
            boxes = self.detect_faces(img)
            # Skip images with 0 or multiple faces for simplicity
            if len(boxes) == 1:
                emb = self.encode_faces_batch(img, boxes)
                if emb.shape[0] == 1:
                    collected_embeddings.append(emb[0])

        if not collected_embeddings:
            return f"[Registration] No valid single-face images found for '{user_id}'."

        if user_id not in self.user_data:
            self.user_data[user_id] = []
        self.user_data[user_id].extend(collected_embeddings)

        self._save_user_data()
        return f"[Registration] User '{user_id}' registered with {len(collected_embeddings)} embedding(s)."

    def identify_user(self, pil_image, threshold=0.6):
        """
        Identify faces in an image.
        Returns a list of dicts: [{'box':(x1,y1,x2,y2), 'user_id':..., 'similarity':...}, ...]
        """
        boxes = self.detect_faces(pil_image)
        if not boxes:
            return []

        embeddings = self.encode_faces_batch(pil_image, boxes)
        results = []

        for i, emb in enumerate(embeddings):
            best_match = None
            best_sim = 0.0

            for user_id, stored_embs in self.user_data.items():
                # Ensure same shape
                valid_embs = [e for e in stored_embs if e.shape == emb.shape]
                if not valid_embs:
                    continue

                sims = cosine_similarity([emb], valid_embs)
                max_sim = np.max(sims)
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_match = user_id

            if best_sim >= threshold:
                results.append({
                    'box': boxes[i],
                    'user_id': best_match,
                    'similarity': float(best_sim)
                })
            else:
                results.append({
                    'box': boxes[i],
                    'user_id': 'Unknown',
                    'similarity': float(best_sim)
                })

        return results
