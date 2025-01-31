import os
import sys
import math
import requests
import numpy as np
import cv2
import torch
import pickle
import logging
from PIL import Image
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter
import io

import gradio as gr

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from deep_sort_realtime.deepsort_tracker import DeepSort

import mediapipe as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('face_pipeline.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('deep_sort_realtime').setLevel(logging.ERROR)

DEFAULT_MODEL_URL = "https://github.com/wuhplaptop/face-11-n/blob/main/face2.pt?raw=true"
DEFAULT_DB_PATH = os.path.expanduser("~/.face_pipeline/known_faces.pkl")
MODEL_DIR = os.path.expanduser("~/.face_pipeline/models")
CONFIG_PATH = os.path.expanduser("~/.face_pipeline/config.pkl")

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

@dataclass
class PipelineConfig:
    detector: Dict = field(default_factory=dict)
    tracker: Dict = field(default_factory=dict)
    recognition: Dict = field(default_factory=dict)
    anti_spoof: Dict = field(default_factory=dict)
    blink: Dict = field(default_factory=dict)
    face_mesh_options: Dict = field(default_factory=dict)
    hand: Dict = field(default_factory=dict)
    eye_color: Dict = field(default_factory=dict)
    enabled_components: Dict = field(default_factory=dict)

    detection_conf_thres: float = 0.4
    recognition_conf_thres: float = 0.85

    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    spoofed_bbox_color: Tuple[int, int, int] = (0, 0, 255)
    unknown_bbox_color: Tuple[int, int, int] = (0, 0, 255)
    eye_outline_color: Tuple[int, int, int] = (255, 255, 0)
    blink_text_color: Tuple[int, int, int] = (0, 0, 255)
    hand_landmark_color: Tuple[int, int, int] = (255, 210, 77)
    hand_connection_color: Tuple[int, int, int] = (204, 102, 0)
    hand_text_color: Tuple[int, int, int] = (255, 255, 255)
    mesh_color: Tuple[int, int, int] = (100, 255, 100)
    contour_color: Tuple[int, int, int] = (200, 200, 0)
    iris_color: Tuple[int, int, int] = (255, 0, 255)
    eye_color_text_color: Tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self):
        self.detector = self.detector or {
            'model_path': os.path.join(MODEL_DIR, "face2.pt"),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        self.tracker = self.tracker or {'max_age': 30}
        self.recognition = self.recognition or {'enable': True}
        self.anti_spoof = self.anti_spoof or {'enable': True, 'lap_thresh': 80.0}
        self.blink = self.blink or {'enable': True, 'ear_thresh': 0.25}
        self.face_mesh_options = self.face_mesh_options or {
            'enable': False,
            'tesselation': False,
            'contours': False,
            'irises': False,
        }
        self.hand = self.hand or {
            'enable': True,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
        }
        self.eye_color = self.eye_color or {'enable': False}
        self.enabled_components = self.enabled_components or {
            'detection': True,
            'tracking': True,
            'anti_spoof': True,
            'recognition': True,
            'blink': True,
            'face_mesh': False,
            'hand': True,
            'eye_color': False,
        }

    def save(self, path: str):
        """Save this config to a pickle file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.__dict__, f)
            logger.info(f"Saved config to {path}")
        except Exception as e:
            logger.error(f"Config save failed: {str(e)}")
            raise RuntimeError(f"Config save failed: {str(e)}") from e

    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        """Load a config from a pickle file."""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                return cls(**data)
            return cls()
        except Exception as e:
            logger.error(f"Config load failed: {str(e)}")
            return cls()

    def export_config(self) -> bytes:
        """Export your config to bytes."""
        try:
            config_data = self.__dict__
            buf = io.BytesIO()
            pickle.dump(config_data, buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"Export config failed: {str(e)}")
            raise RuntimeError(f"Export config failed: {str(e)}") from e

    @classmethod
    def import_config(cls, config_bytes: bytes) -> 'PipelineConfig':
        """Import config from bytes."""
        try:
            buf = io.BytesIO(config_bytes)
            data = pickle.load(buf)
            return cls(**data)
        except Exception as e:
            logger.error(f"Import config failed: {str(e)}")
            raise RuntimeError(f"Import config failed: {str(e)}") from e

class FaceDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded database from {self.db_path}")
        except Exception as e:
            logger.error(f"Database load failed: {str(e)}")
            self.embeddings = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Saved database to {self.db_path}")
        except Exception as e:
            logger.error(f"Database save failed: {str(e)}")
            raise RuntimeError(f"Database save failed: {str(e)}") from e

    def export_database(self) -> bytes:
        """Export the entire face embeddings DB to bytes."""
        try:
            db_data = self.embeddings
            buf = io.BytesIO()
            pickle.dump(db_data, buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"Export database failed: {str(e)}")
            raise RuntimeError(f"Export database failed: {str(e)}") from e

    def import_database(self, db_bytes: bytes, merge: bool = True):
        """
        Import embeddings from bytes.
        If merge=True, merges with current DB. If False, overwrites.
        """
        try:
            buf = io.BytesIO(db_bytes)
            imported_data = pickle.load(buf)
            if not isinstance(imported_data, dict):
                raise ValueError("Imported data is not a dictionary!")

            if merge:
                for label, emb_list in imported_data.items():
                    if label not in self.embeddings:
                        self.embeddings[label] = []
                    self.embeddings[label].extend(emb_list)
            else:
                self.embeddings = imported_data

            self.save()
            logger.info(f"Imported face database, merge={merge}")
        except Exception as e:
            logger.error(f"Import database failed: {str(e)}")
            raise RuntimeError(f"Import database failed: {str(e)}") from e

    def add_embedding(self, label: str, embedding: np.ndarray):
        try:
            if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
                raise ValueError("Invalid embedding format")
            if label not in self.embeddings:
                self.embeddings[label] = []
            self.embeddings[label].append(embedding)
            logger.debug(f"Added embedding for {label}")
        except Exception as e:
            logger.error(f"Add embedding failed: {str(e)}")
            raise

    def remove_label(self, label: str):
        try:
            if label in self.embeddings:
                del self.embeddings[label]
                logger.info(f"Removed {label}")
            else:
                logger.warning(f"Label {label} not found")
        except Exception as e:
            logger.error(f"Remove label failed: {str(e)}")
            raise

    def list_labels(self) -> List[str]:
        return list(self.embeddings.keys())

    def get_embeddings_by_label(self, label: str) -> Optional[List[np.ndarray]]:
        return self.embeddings.get(label)

    def search_by_image(self, query_embedding: np.ndarray, threshold: float = 0.7) -> List[Tuple[str, float]]:
        results = []
        for lbl, embs in self.embeddings.items():
            for db_emb in embs:
                sim = FacePipeline.cosine_similarity(query_embedding, db_emb)
                if sim >= threshold:
                    results.append((lbl, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

class YOLOFaceDetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = None
        self.device = device
        try:
            if not os.path.exists(model_path):
                logger.info(f"Model not found at {model_path}. Downloading from GitHub...")
                resp = requests.get(DEFAULT_MODEL_URL)
                resp.raise_for_status()
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(resp.content)
                logger.info(f"Downloaded YOLO model to {model_path}")

            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"YOLO init failed: {str(e)}")
            raise

    def detect(self, image: np.ndarray, conf_thres: float) -> List[Tuple[int, int, int, int, float, int]]:
        try:
            results = self.model.predict(
                source=image, conf=conf_thres, verbose=False, device=self.device
            )
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
            logger.debug(f"Detected {len(detections)} faces.")
            return detections
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []

class FaceTracker:
    def __init__(self, max_age: int = 30):
        self.tracker = DeepSort(max_age=max_age, embedder='mobilenet')

    def update(self, detections: List[Tuple], frame: np.ndarray):
        try:
            ds_detections = [
                ([x1, y1, x2 - x1, y2 - y1], conf, cls)
                for (x1, y1, x2, y2, conf, cls) in detections
            ]
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
            logger.debug(f"Updated tracker with {len(tracks)} tracks.")
            return tracks
        except Exception as e:
            logger.error(f"Tracking error: {str(e)}")
            return []

class FaceNetEmbedder:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb).convert('RGB')
            tens = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(tens)[0].cpu().numpy()
            logger.debug(f"Generated embedding sample: {embedding[:5]}...")
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            return None

def detect_blink(face_roi: np.ndarray, threshold: float = 0.25) -> Tuple[bool, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      (blink_bool, left_ear, right_ear, left_eye_points, right_eye_points).
    """
    try:
        face_mesh_proc = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        result = face_mesh_proc.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        face_mesh_proc.close()

        if not result.multi_face_landmarks:
            return False, 0.0, 0.0, None, None

        landmarks = result.multi_face_landmarks[0].landmark
        h, w = face_roi.shape[:2]

        def eye_aspect_ratio(indices):
            pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
            vertical = np.linalg.norm(np.array(pts[1]) - np.array(pts[5])) + \
                       np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            horizontal = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            return vertical / (2.0 * horizontal + 1e-6)

        left_ear = eye_aspect_ratio(LEFT_EYE_IDX)
        right_ear = eye_aspect_ratio(RIGHT_EYE_IDX)

        blink = (left_ear < threshold) and (right_ear < threshold)

        left_eye_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX])
        right_eye_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX])

        return blink, left_ear, right_ear, left_eye_pts, right_eye_pts

    except Exception as e:
        logger.error(f"Blink detection error: {str(e)}")
        return False, 0.0, 0.0, None, None

def process_face_mesh(face_roi: np.ndarray):
    try:
        fm_proc = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        result = fm_proc.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        fm_proc.close()
        if result.multi_face_landmarks:
            return result.multi_face_landmarks[0]
        return None
    except Exception as e:
        logger.error(f"Face mesh error: {str(e)}")
        return None

def draw_face_mesh(image: np.ndarray, face_landmarks, config: Dict, pipeline_config: PipelineConfig):
    mesh_color_bgr = pipeline_config.mesh_color[::-1]
    contour_color_bgr = pipeline_config.contour_color[::-1]
    iris_color_bgr = pipeline_config.iris_color[::-1]

    if config.get('tesselation'):
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=mesh_color_bgr, thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=mesh_color_bgr, thickness=1),
        )
    if config.get('contours'):
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=contour_color_bgr, thickness=2)
        )
    if config.get('irises'):
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=iris_color_bgr, thickness=2)
        )

EYE_COLOR_RANGES = {
    "amber": (255, 191, 0),
    "blue": (0, 0, 255),
    "brown": (139, 69, 19),
    "green": (0, 128, 0),
    "gray": (128, 128, 128),
    "hazel": (102, 51, 0),
}

def classify_eye_color(rgb_color: Tuple[int,int,int]) -> str:
    if rgb_color is None:
        return "Unknown"
    min_dist = float('inf')
    best = "Unknown"
    for color_name, ref_rgb in EYE_COLOR_RANGES.items():
        dist = math.sqrt(sum([(a-b)**2 for a,b in zip(rgb_color, ref_rgb)]))
        if dist < min_dist:
            min_dist = dist
            best = color_name
    return best

def get_dominant_color(image_roi, k=3):
    if image_roi.size == 0:
        return None
    pixels = np.float32(image_roi.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dom_color = tuple(palette[np.argmax(counts)].astype(int).tolist())
    return dom_color

def detect_eye_color(face_roi: np.ndarray, face_landmarks) -> Optional[str]:
    if face_landmarks is None:
        return None
    h, w = face_roi.shape[:2]
    iris_inds = set()
    for conn in mp_face_mesh.FACEMESH_IRISES:
        iris_inds.update(conn)

    iris_points = []
    for idx in iris_inds:
        lm = face_landmarks.landmark[idx]
        iris_points.append((int(lm.x * w), int(lm.y * h)))
    if not iris_points:
        return None

    min_x = min(pt[0] for pt in iris_points)
    max_x = max(pt[0] for pt in iris_points)
    min_y = min(pt[1] for pt in iris_points)
    max_y = max(pt[1] for pt in iris_points)

    pad = 5
    x1 = max(0, min_x - pad)
    y1 = max(0, min_y - pad)
    x2 = min(w, max_x + pad)
    y2 = min(h, max_y + pad)

    eye_roi = face_roi[y1:y2, x1:x2]
    eye_roi_resize = cv2.resize(eye_roi, (40, 40), interpolation=cv2.INTER_AREA)

    if eye_roi_resize.size == 0:
        return None

    dom_rgb = get_dominant_color(eye_roi_resize)
    if dom_rgb is not None:
        return classify_eye_color(dom_rgb)
    return None

class HandTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info("Initialized Mediapipe HandTracking")

    def detect_hands(self, image: np.ndarray):
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            return results.multi_hand_landmarks, results.multi_handedness
        except Exception as e:
            logger.error(f"Hand detection error: {str(e)}")
            return None, None

    def draw_hands(self, image: np.ndarray, hand_landmarks, handedness, config: Dict):
        if not hand_landmarks:
            return image

        for i, hlms in enumerate(hand_landmarks):
            hl_color = config.hand_landmark_color[::-1]
            hc_color = config.hand_connection_color[::-1]
            mp_drawing.draw_landmarks(
                image,
                hlms,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=hl_color, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=hc_color, thickness=2, circle_radius=2),
            )
            if handedness and i < len(handedness):
                label = handedness[i].classification[0].label
                score = handedness[i].classification[0].score
                text = f"{label}: {score:.2f}"

                wrist_lm = hlms.landmark[mp_hands.HandLandmark.WRIST]
                h, w_img, _ = image.shape
                cx, cy = int(wrist_lm.x * w_img), int(wrist_lm.y * h)
                ht_color = config.hand_text_color[::-1]
                cv2.putText(image, text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ht_color, 2)
        return image

class FacePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = None
        self.tracker = None
        self.facenet = None
        self.db = None
        self.hand_tracker = None
        self._initialized = False

    def initialize(self):
        try:
            self.detector = YOLOFaceDetector(
                model_path=self.config.detector['model_path'],
                device=self.config.detector['device']
            )
            self.tracker = FaceTracker(max_age=self.config.tracker['max_age'])
            self.facenet = FaceNetEmbedder(device=self.config.detector['device'])
            self.db = FaceDatabase()

            if self.config.hand['enable']:
                self.hand_tracker = HandTracker(
                    min_detection_confidence=self.config.hand['min_detection_confidence'],
                    min_tracking_confidence=self.config.hand['min_tracking_confidence']
                )

            self._initialized = True
            logger.info("FacePipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            self._initialized = False
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Main pipeline processing: detection, tracking, hand detection, face mesh, blink detection, etc.
        Returns annotated_frame, detection_results.
        """
        if not self._initialized:
            logger.error("Pipeline not initialized.")
            return frame, []

        try:
            detections = self.detector.detect(frame, self.config.detection_conf_thres)
            tracked_objs = self.tracker.update(detections, frame)
            annotated = frame.copy()
            results = []

            # Hand detection
            hand_landmarks_list = None
            handedness_list = None
            if self.config.hand['enable'] and self.hand_tracker:
                hand_landmarks_list, handedness_list = self.hand_tracker.detect_hands(annotated)
                annotated = self.hand_tracker.draw_hands(
                    annotated, hand_landmarks_list, handedness_list, self.config
                )

            for obj in tracked_objs:
                if not obj.is_confirmed():
                    continue

                track_id = obj.track_id
                bbox = obj.to_tlbr().astype(int)
                x1, y1, x2, y2 = bbox
                conf = getattr(obj, 'score', 1.0)
                cls = getattr(obj, 'class_id', 0)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    logger.warning(f"Empty face ROI for track={track_id}")
                    continue

                # Anti-spoof
                is_spoofed = False
                if self.config.anti_spoof.get('enable', True):
                    is_spoofed = not self.is_real_face(face_roi)
                    if is_spoofed:
                        cls = 1  # Mark as "spoof"

                if is_spoofed:
                    box_color_bgr = self.config.spoofed_bbox_color[::-1]
                    name = "Spoofed"
                    similarity = 0.0
                else:
                    # Face recognition
                    emb = self.facenet.get_embedding(face_roi)
                    if emb is not None and self.config.recognition.get('enable', True):
                        name, similarity = self.recognize_face(emb, self.config.recognition_conf_thres)
                    else:
                        name = "Unknown"
                        similarity = 0.0

                    box_color_rgb = (self.config.bbox_color if name != "Unknown"
                                     else self.config.unknown_bbox_color)
                    box_color_bgr = box_color_rgb[::-1]

                label_text = name
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color_bgr, 2)
                cv2.putText(annotated, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_bgr, 2)

                # Blink detection
                blink = False
                if self.config.blink.get('enable', False):
                    blink, left_ear, right_ear, left_eye_pts, right_eye_pts = detect_blink(
                        face_roi, threshold=self.config.blink.get('ear_thresh', 0.25)
                    )
                    if left_eye_pts is not None and right_eye_pts is not None:
                        le_g = left_eye_pts + np.array([x1, y1])
                        re_g = right_eye_pts + np.array([x1, y1])

                        eye_outline_bgr = self.config.eye_outline_color[::-1]
                        cv2.polylines(annotated, [le_g], True, eye_outline_bgr, 1)
                        cv2.polylines(annotated, [re_g], True, eye_outline_bgr, 1)
                        if blink:
                            blink_msg_color = self.config.blink_text_color[::-1]
                            cv2.putText(annotated, "Blink Detected",
                                        (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        blink_msg_color, 2)

                # Face mesh
                face_mesh_landmarks = None
                eye_color_name = None
                if (self.config.face_mesh_options.get('enable') or
                        self.config.eye_color.get('enable')):
                    face_mesh_landmarks = process_face_mesh(face_roi)
                    if face_mesh_landmarks:
                        # Draw mesh
                        if self.config.face_mesh_options.get('enable', False):
                            draw_face_mesh(
                                annotated[y1:y2, x1:x2],
                                face_mesh_landmarks,
                                self.config.face_mesh_options,
                                self.config
                            )

                        # Eye color detection
                        if self.config.eye_color.get('enable', False):
                            color_found = detect_eye_color(face_roi, face_mesh_landmarks)
                            if color_found:
                                eye_color_name = color_found
                                text_col_bgr = self.config.eye_color_text_color[::-1]
                                cv2.putText(
                                    annotated, f"Eye Color: {eye_color_name}",
                                    (x1, y2 + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    text_col_bgr, 2
                                )

                detection_info = {
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "class_id": cls,
                    "name": name,
                    "similarity": similarity,
                    "blink": blink if self.config.blink.get('enable') else None,
                    "face_mesh": bool(face_mesh_landmarks) if self.config.face_mesh_options.get('enable') else False,
                    "hands_detected": bool(hand_landmarks_list),
                    "hand_count": len(hand_landmarks_list) if hand_landmarks_list else 0,
                    "eye_color": eye_color_name if self.config.eye_color.get('enable') else None
                }
                results.append(detection_info)

            return annotated, results

        except Exception as e:
            logger.error(f"Frame process error: {str(e)}")
            return frame, []

    def is_real_face(self, face_roi: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            lapv = cv2.Laplacian(gray, cv2.CV_64F).var()
            return lapv > self.config.anti_spoof.get('lap_thresh', 80.0)
        except Exception as e:
            logger.error(f"Anti-spoof error: {str(e)}")
            return False

    def recognize_face(self, embedding: np.ndarray, threshold: float) -> Tuple[str, float]:
        try:
            best_name = "Unknown"
            best_sim = 0.0
            for lbl, embs in self.db.embeddings.items():
                for db_emb in embs:
                    sim = FacePipeline.cosine_similarity(embedding, db_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = lbl
            if best_sim < threshold:
                best_name = "Unknown"
            return best_name, best_sim
        except Exception as e:
            logger.error(f"Recognition error: {str(e)}")
            return ("Unknown", 0.0)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-6))

pipeline = None
def load_pipeline() -> FacePipeline:
    """Global pipeline loader. Creates if not exists, or returns existing one."""
    global pipeline
    if pipeline is None:
        cfg = PipelineConfig.load(CONFIG_PATH)
        pipeline = FacePipeline(cfg)
        pipeline.initialize()
    return pipeline

def hex_to_bgr(hexstr: str) -> Tuple[int,int,int]:
    if not hexstr.startswith('#'):
        hexstr = '#' + hexstr
    h = hexstr.lstrip('#')
    if len(h) != 6:
        return (255, 0, 0)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b,g,r)

def bgr_to_hex(bgr: Tuple[int,int,int]) -> str:
    b,g,r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

def update_config(
    enable_recognition, enable_antispoof, enable_blink, enable_hand, enable_eyecolor, enable_facemesh,
    show_tesselation, show_contours, show_irises,
    detection_conf, recognition_thresh, antispoof_thresh, blink_thresh, hand_det_conf, hand_track_conf,
    bbox_hex, spoofed_hex, unknown_hex, eye_hex, blink_hex,
    hand_landmark_hex, hand_connect_hex, hand_text_hex,
    mesh_hex, contour_hex, iris_hex, eye_color_text_hex
):
    pl = load_pipeline()
    cfg = pl.config

    cfg.recognition['enable'] = enable_recognition
    cfg.anti_spoof['enable'] = enable_antispoof
    cfg.blink['enable'] = enable_blink
    cfg.hand['enable'] = enable_hand
    cfg.eye_color['enable'] = enable_eyecolor
    cfg.face_mesh_options['enable'] = enable_facemesh

    cfg.face_mesh_options['tesselation'] = show_tesselation
    cfg.face_mesh_options['contours'] = show_contours
    cfg.face_mesh_options['irises'] = show_irises

    cfg.detection_conf_thres = detection_conf
    cfg.recognition_conf_thres = recognition_thresh
    cfg.anti_spoof['lap_thresh'] = antispoof_thresh
    cfg.blink['ear_thresh'] = blink_thresh
    cfg.hand['min_detection_confidence'] = hand_det_conf
    cfg.hand['min_tracking_confidence'] = hand_track_conf

    cfg.bbox_color = hex_to_bgr(bbox_hex)[::-1]
    cfg.spoofed_bbox_color = hex_to_bgr(spoofed_hex)[::-1]
    cfg.unknown_bbox_color = hex_to_bgr(unknown_hex)[::-1]
    cfg.eye_outline_color = hex_to_bgr(eye_hex)[::-1]
    cfg.blink_text_color = hex_to_bgr(blink_hex)[::-1]
    cfg.hand_landmark_color = hex_to_bgr(hand_landmark_hex)[::-1]
    cfg.hand_connection_color = hex_to_bgr(hand_connect_hex)[::-1]
    cfg.hand_text_color = hex_to_bgr(hand_text_hex)[::-1]
    cfg.mesh_color = hex_to_bgr(mesh_hex)[::-1]
    cfg.contour_color = hex_to_bgr(contour_hex)[::-1]
    cfg.iris_color = hex_to_bgr(iris_hex)[::-1]
    cfg.eye_color_text_color = hex_to_bgr(eye_color_text_hex)[::-1]

    cfg.save(CONFIG_PATH)
    return "Configuration saved successfully!"

def enroll_user(label_name: str, files: List[bytes]) -> str:
    """Enrolls a user by name using multiple uploaded image files."""
    pl = load_pipeline()
    if not label_name:
        return "Please provide a user name."
    if not files or len(files) == 0:
        return "No images provided."

    enrolled_count = 0
    for file_bytes in files:
        if not file_bytes:
            continue
        try:
            img_array = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            dets = pl.detector.detect(img_bgr, pl.config.detection_conf_thres)
            for x1, y1, x2, y2, conf, cls in dets:
                roi = img_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                emb = pl.facenet.get_embedding(roi)
                if emb is not None:
                    pl.db.add_embedding(label_name, emb)
                    enrolled_count += 1
        except Exception as e:
            logger.error(f"Error enrolling user from file: {str(e)}")
            continue

    if enrolled_count > 0:
        pl.db.save()
        return f"Enrolled '{label_name}' with {enrolled_count} face(s)!"
    else:
        return "No faces detected in provided images."

def search_by_name(name: str) -> str:
    pl = load_pipeline()
    if not name:
        return "No name entered."
    embs = pl.db.get_embeddings_by_label(name)
    if embs:
        return f"'{name}' found with {len(embs)} embedding(s)."
    else:
        return f"No embeddings found for '{name}'."

def search_by_image(img: np.ndarray) -> str:
    pl = load_pipeline()
    if img is None:
        return "No image uploaded."
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dets = pl.detector.detect(img_bgr, pl.config.detection_conf_thres)
    if not dets:
        return "No faces detected in the uploaded image."
    x1, y1, x2, y2, conf, cls = dets[0]
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "Empty face ROI in the uploaded image."

    emb = pl.facenet.get_embedding(roi)
    if emb is None:
        return "Could not generate embedding from face."
    results = pl.db.search_by_image(emb, pl.config.recognition_conf_thres)
    if not results:
        return "No matches in the database under current threshold."
    lines = [f"- {lbl} (sim={sim:.3f})" for lbl, sim in results]
    return "Search results:\n" + "\n".join(lines)

def remove_user(label: str) -> str:
    pl = load_pipeline()
    if not label:
        return "No user label selected."
    pl.db.remove_label(label)
    pl.db.save()
    return f"User '{label}' removed."

def list_users() -> str:
    pl = load_pipeline()
    labels = pl.db.list_labels()
    if labels:
        return "Enrolled users:\n" + ", ".join(labels)
    return "No users enrolled."

def process_test_image(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """Single-image test: run pipeline and return annotated image + JSON results."""
    if img is None:
        return None, "No image uploaded."
    pl = load_pipeline()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    processed, detections = pl.process_frame(bgr)
    result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return result_rgb, str(detections)

# ===================================
# Combined Export/Import (Config + DB)
# ===================================
def export_all_file() -> Tuple[bytes, str]:
    """
    Exports both the pipeline config and database embeddings into a single
    pickle file. Returns the file content and filename as a tuple for Gradio to handle the download.
    """
    pl = load_pipeline()
    combined_data = {
        "config": pl.config.__dict__,
        "database": pl.db.embeddings
    }

    # Create an in-memory buffer and pickle the combined data
    buf = io.BytesIO()
    pickle.dump(combined_data, buf)
    buf.seek(0)

    # Read the buffer's content
    file_content = buf.read()

    # Return a tuple of (file content, filename)
    return (file_content, "pipeline_export.pkl")

def import_all_file(file_bytes: bytes, merge_db: bool = True) -> str:
    """
    Imports a single pickle file containing both the config and database.
    If merge_db=False, overwrites the existing DB; otherwise merges.
    """
    if file_bytes is None:
        return "No file provided."

    try:
        # Load the data from the bytes
        buf = io.BytesIO(file_bytes)
        combined_data = pickle.load(buf)

        if not isinstance(combined_data, dict):
            return "Invalid combined data format."

        # Rebuild config
        new_cfg_data = combined_data.get("config", {})
        new_cfg = PipelineConfig(**new_cfg_data)

        # Rebuild DB
        new_db_data = combined_data.get("database", {})

        # Re-initialize pipeline with new config
        global pipeline
        pipeline = FacePipeline(new_cfg)
        pipeline.initialize()

        # Merge or overwrite DB
        if merge_db:
            # Merge
            for label, emb_list in new_db_data.items():
                if label not in pipeline.db.embeddings:
                    pipeline.db.embeddings[label] = []
                pipeline.db.embeddings[label].extend(emb_list)
            pipeline.db.save()
        else:
            # Overwrite
            pipeline.db.embeddings = new_db_data
            pipeline.db.save()

        return "Config and database imported successfully!"

    except Exception as e:
        logger.error(f"Import all failed: {str(e)}")
        return f"Import failed: {str(e)}"

# ==========================
# Original Export/Import for
# Config or DB individually
# ==========================

def export_config_file() -> Tuple[bytes, str]:
    """Export the current pipeline config as a downloadable file."""
    pl = load_pipeline()
    config_bytes = pl.config.export_config()
    return (config_bytes, "config_export.pkl")

def import_config_file(file_bytes: bytes) -> str:
    """Import a pipeline config from uploaded bytes and re-initialize pipeline."""
    if not file_bytes:
        return "No file provided."
    try:
        new_cfg = PipelineConfig.import_config(file_bytes)
        pl = FacePipeline(new_cfg)
        pl.initialize()
        global pipeline
        pipeline = pl
        return f"Imported config successfully!"
    except Exception as e:
        logger.error(f"Import config failed: {str(e)}")
        return f"Import failed: {str(e)}"

def export_db_file() -> Tuple[bytes, str]:
    """Export the current face database as a downloadable file."""
    pl = load_pipeline()
    db_bytes = pl.db.export_database()
    return (db_bytes, "database_export.pkl")

def import_db_file(db_bytes: bytes, merge: bool=True) -> str:
    """Import face database from uploaded bytes. Merge or overwrite existing."""
    if not db_bytes:
        return "No file provided."
    try:
        pl = load_pipeline()
        pl.db.import_database(db_bytes, merge=merge)
        return f"Database imported successfully, merge={merge}"
    except Exception as e:
        logger.error(f"Import DB failed: {str(e)}")
        return f"Import DB failed: {str(e)}"

# Build Gradio App
def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# FaceRec: Comprehensive Face Recognition Pipeline")

        with gr.Tab("Image Test"):
            gr.Markdown("Upload a single image to detect faces, run blink detection, face mesh, hand tracking, etc.")
            test_in = gr.Image(type="numpy", label="Upload Image")
            test_out = gr.Image()
            test_info = gr.Textbox(label="Detections")
            process_btn = gr.Button("Process Image")

            process_btn.click(
                fn=process_test_image,
                inputs=test_in,
                outputs=[test_out, test_info],
            )

        with gr.Tab("Configuration"):
            gr.Markdown("Adjust toggles, thresholds, and colors. Click Save to persist changes.")

            with gr.Row():
                enable_recognition = gr.Checkbox(label="Enable Recognition", value=True)
                enable_antispoof = gr.Checkbox(label="Enable Anti-Spoof", value=True)
                enable_blink = gr.Checkbox(label="Enable Blink Detection", value=True)
                enable_hand = gr.Checkbox(label="Enable Hand Tracking", value=True)
                enable_eyecolor = gr.Checkbox(label="Enable Eye Color Detection", value=False)
                enable_facemesh = gr.Checkbox(label="Enable Face Mesh", value=False)

            gr.Markdown("**Face Mesh Options**")
            with gr.Row():
                show_tesselation = gr.Checkbox(label="Tesselation", value=False)
                show_contours = gr.Checkbox(label="Contours", value=False)
                show_irises = gr.Checkbox(label="Irises", value=False)

            gr.Markdown("**Thresholds**")
            detection_conf = gr.Slider(0, 1, 0.4, step=0.01, label="Detection Confidence")
            recognition_thresh = gr.Slider(0.5, 1.0, 0.85, step=0.01, label="Recognition Threshold")
            antispoof_thresh = gr.Slider(0, 200, 80, step=1, label="Anti-Spoof Threshold")
            blink_thresh = gr.Slider(0, 0.5, 0.25, step=0.01, label="Blink EAR Threshold")
            hand_det_conf = gr.Slider(0, 1, 0.5, step=0.01, label="Hand Detection Confidence")
            hand_track_conf = gr.Slider(0, 1, 0.5, step=0.01, label="Hand Tracking Confidence")

            gr.Markdown("**Color Options (Hex)**")
            bbox_hex = gr.Textbox(label="Box Color (Recognized)", value="#00ff00")
            spoofed_hex = gr.Textbox(label="Box Color (Spoofed)", value="#ff0000")
            unknown_hex = gr.Textbox(label="Box Color (Unknown)", value="#ff0000")
            eye_hex = gr.Textbox(label="Eye Outline Color", value="#ffff00")
            blink_hex = gr.Textbox(label="Blink Text Color", value="#0000ff")

            hand_landmark_hex = gr.Textbox(label="Hand Landmark Color", value="#ffd24d")
            hand_connect_hex = gr.Textbox(label="Hand Connection Color", value="#cc6600")
            hand_text_hex = gr.Textbox(label="Hand Text Color", value="#ffffff")

            mesh_hex = gr.Textbox(label="Mesh Color", value="#64ff64")
            contour_hex = gr.Textbox(label="Contour Color", value="#c8c800")
            iris_hex = gr.Textbox(label="Iris Color", value="#ff00ff")
            eye_color_text_hex = gr.Textbox(label="Eye Color Text Color", value="#ffffff")

            save_btn = gr.Button("Save Configuration")
            save_msg = gr.Textbox(label="", interactive=False)

            save_btn.click(
                fn=update_config,
                inputs=[
                    enable_recognition, enable_antispoof, enable_blink, enable_hand, enable_eyecolor, enable_facemesh,
                    show_tesselation, show_contours, show_irises,
                    detection_conf, recognition_thresh, antispoof_thresh, blink_thresh, hand_det_conf, hand_track_conf,
                    bbox_hex, spoofed_hex, unknown_hex, eye_hex, blink_hex,
                    hand_landmark_hex, hand_connect_hex, hand_text_hex,
                    mesh_hex, contour_hex, iris_hex, eye_color_text_hex
                ],
                outputs=[save_msg]
            )

        with gr.Tab("Database Management"):
            gr.Markdown("Enroll multiple images per user, search by name or image, remove users, list all users.")

            with gr.Accordion("User Enrollment", open=False):
                enroll_name = gr.Textbox(label="User Name")
                enroll_paths = gr.File(file_count="multiple", type="binary", label="Upload Multiple Images")  # Updated
                enroll_btn = gr.Button("Enroll User")
                enroll_result = gr.Textbox()

                enroll_btn.click(
                    fn=enroll_user,
                    inputs=[enroll_name, enroll_paths],
                    outputs=[enroll_result]
                )

            with gr.Accordion("User Search", open=False):
                search_mode = gr.Radio(["Name", "Image"], label="Search By", value="Name")
                search_name_box = gr.Dropdown(label="Select User", choices=[], value=None, visible=True)
                search_image_box = gr.Image(label="Upload Search Image", type="numpy", visible=False)
                search_btn = gr.Button("Search")
                search_out = gr.Textbox()

                def toggle_search(mode):
                    if mode == "Name":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)

                search_mode.change(
                    fn=toggle_search,
                    inputs=[search_mode],
                    outputs=[search_name_box, search_image_box]
                )

                def do_search(mode, uname, img):
                    if mode == "Name":
                        return search_by_name(uname)
                    else:
                        return search_by_image(img)

                search_btn.click(
                    fn=do_search,
                    inputs=[search_mode, search_name_box, search_image_box],
                    outputs=[search_out]
                )

            with gr.Accordion("User Management Tools", open=False):
                list_btn = gr.Button("List Enrolled Users")
                list_out = gr.Textbox()
                list_btn.click(fn=lambda: list_users(), inputs=[], outputs=[list_out])

                def refresh_choices():
                    pl = load_pipeline()
                    return gr.update(choices=pl.db.list_labels())

                refresh_btn = gr.Button("Refresh User List")
                refresh_btn.click(fn=refresh_choices, inputs=[], outputs=[search_name_box])

                remove_box = gr.Dropdown(label="Select User to Remove", choices=[])
                remove_btn = gr.Button("Remove")
                remove_out = gr.Textbox()

                remove_btn.click(fn=remove_user, inputs=[remove_box], outputs=[remove_out])
                refresh_btn.click(fn=refresh_choices, inputs=[], outputs=[remove_box])

        with gr.Tab("Export / Import"):
            gr.Markdown("Export or import pipeline config (thresholds/colors) or face database (embeddings).")

            gr.Markdown("**Export Individually (Download)**")
            export_config_btn = gr.Button("Export Config")
            export_config_download = gr.Download(label="Download Config Export")  # Changed to gr.Download

            export_db_btn = gr.Button("Export Database")
            export_db_download = gr.Download(label="Download Database Export")  # Changed to gr.Download

            export_config_btn.click(export_config_file, inputs=[], outputs=[export_config_download])
            export_db_btn.click(export_db_file, inputs=[], outputs=[export_db_download])

            gr.Markdown("**Import Individually (Upload)**")
            import_config_filebox = gr.File(label="Import Config File", file_count="single", type="binary")  # Updated
            import_config_btn = gr.Button("Import Config")
            import_config_out = gr.Textbox()

            import_db_filebox = gr.File(label="Import Database File", file_count="single", type="binary")  # Updated
            merge_db_checkbox = gr.Checkbox(label="Merge instead of overwrite?", value=True)
            import_db_btn = gr.Button("Import Database")
            import_db_out = gr.Textbox()

            import_config_btn.click(fn=import_config_file, inputs=[import_config_filebox], outputs=[import_config_out])
            import_db_btn.click(fn=import_db_file, inputs=[import_db_filebox, merge_db_checkbox], outputs=[import_db_out])

            # =============================
            # Export/Import All Together
            # =============================
            gr.Markdown("---")
            gr.Markdown("**Export & Import Everything (Config + Database) Together**")

            # For exporting: we'll just produce a file in-memory
            export_all_btn = gr.Button("Export All (Config + DB)")
            export_all_download = gr.Download(label="Download Combined Export")  # Changed to gr.Download

            export_all_btn.click(
                fn=export_all_file,
                outputs=[export_all_download],
                inputs=[]
            )

            # For importing: user uploads file
            import_all_in = gr.File(label="Import Combined File (Pickle)", file_count="single", type="binary")  # Updated
            import_all_merge_cb = gr.Checkbox(label="Merge DB instead of overwrite?", value=True)
            import_all_btn = gr.Button("Import All")
            import_all_out = gr.Textbox()

            import_all_btn.click(
                fn=import_all_file,
                inputs=[import_all_in, import_all_merge_cb],
                outputs=[import_all_out]
            )

        return demo

def main():
    """Entry point to launch the Gradio app."""
    app = build_app()
    # We add `.queue()` so that multiple requests can be queued
    app.queue().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
