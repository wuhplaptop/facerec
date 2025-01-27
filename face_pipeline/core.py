# face_pipeline/core.py

import os
import logging
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import mediapipe as mp

from .config import PipelineConfig
from .database import FaceDatabase
from .detectors import YOLOFaceDetector
from .trackers import FaceTracker
from .embedder import FaceNetEmbedder
from .utilities import (
    detect_blink,
    classify_eye_color,
    get_dominant_color,
    process_face_mesh,
    draw_face_mesh,
    detect_eye_color,
    HandTracker,
)

logger = logging.getLogger(__name__)

@dataclass
class FacePipeline:
    config: PipelineConfig
    detector: YOLOFaceDetector = field(init=False)
    tracker: FaceTracker = field(init=False)
    facenet: FaceNetEmbedder = field(init=False)
    db: FaceDatabase = field(init=False)
    hand_tracker: HandTracker = field(init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self):
        try:
            self.detector = YOLOFaceDetector(
                model_path=self.config.detector['model_path'],
                device=self.config.detector['device']
            )
            self.tracker = FaceTracker(max_age=self.config.tracker['max_age'])
            self.facenet = FaceNetEmbedder(device=self.config.detector['device'])
            self.db = FaceDatabase(db_path=self.config.DEFAULT_DB_PATH)

            if self.config.hand['enable']:
                self.hand_tracker = HandTracker(
                    min_detection_confidence=self.config.hand['min_detection_confidence'],
                    min_tracking_confidence=self.config.hand['min_tracking_confidence'],
                )

            logger.info("FacePipeline initialized successfully.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            self._initialized = False
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        if not self._initialized:
            logger.error("Pipeline not initialized!")
            return frame, []

        try:
            detections = self.detector.detect(frame, self.config.detection_conf_thres)
            tracked_objects = self.tracker.update(detections, frame)
            annotated_frame = frame.copy()
            results = []

            hand_landmarks_list = None
            handedness_results = None

            if self.config.hand['enable']:
                hand_landmarks_list, handedness_results = self.hand_tracker.detect_hands(frame)
                annotated_frame = self.hand_tracker.draw_hands(
                    annotated_frame, hand_landmarks_list, handedness_results, self.config
                )

            for obj in tracked_objects:
                if not obj.is_confirmed():
                    continue
                track_id = obj.track_id
                bbox = obj.to_tlbr()
                x1, y1, x2, y2 = bbox.astype(int)
                conf = obj.score if hasattr(obj, 'score') else 1.0
                cls = obj.class_id if hasattr(obj, 'class_id') else 0

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    logger.warning(f"Empty face ROI for track ID {track_id}. Skipping.")
                    continue

                is_spoofed = False
                if self.config.anti_spoof['enable']:
                    is_spoofed = not self.is_real_face(face_roi)
                    if is_spoofed:
                        logger.info(f"Anti-spoofing check failed for track ID {track_id}.")
                        cls = 1  # Mark as spoofed class

                if is_spoofed:
                    box_color_bgr = self.config.spoofed_bbox_color[::-1]
                    label_text = "Spoofed"
                    name = "Spoofed"
                    similarity = 0.0
                else:
                    embedding = self.facenet.get_embedding(face_roi)
                    if embedding is not None and self.config.recognition['enable']:
                        name, similarity = self.recognize_face(
                            embedding, self.config.recognition_conf_thres
                        )
                    else:
                        name = "Unknown"
                        similarity = 0.0
                    box_color_rgb = self.config.bbox_color if name != "Unknown" else self.config.unknown_bbox_color
                    box_color_bgr = box_color_rgb[::-1]
                    label_text = f"{name} ({similarity:.2f})" if name != "Unknown" else "Unknown"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color_bgr, 2)
                cv2.putText(
                    annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_bgr, 2
                )

                blink = False
                eye_color_name = None

                if self.config.blink['enable']:
                    blink, left_ear, right_ear, left_eye, right_eye = detect_blink(
                        face_roi, threshold=self.config.blink['ear_thresh']
                    )
                    if left_eye is not None and right_eye is not None:
                        left_eye_global = left_eye + np.array([x1, y1])
                        right_eye_global = right_eye + np.array([x1, y1])
                        eye_color_bgr = self.config.eye_outline_color[::-1]
                        cv2.polylines(annotated_frame, [left_eye_global], True, eye_color_bgr, 1)
                        cv2.polylines(annotated_frame, [right_eye_global], True, eye_color_bgr, 1)
                        if blink:
                            blink_text_color_bgr = self.config.blink_text_color[::-1]
                            cv2.putText(
                                annotated_frame,
                                "Blink Detected",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                blink_text_color_bgr,
                                2,
                            )

                face_mesh_landmarks = None
                if self.config.face_mesh_options['enable'] or self.config.eye_color['enable']:
                    face_mesh_landmarks = process_face_mesh(face_roi)
                    if face_mesh_landmarks:
                        if self.config.face_mesh_options['enable']:
                            draw_face_mesh(
                                annotated_frame[y1:y2, x1:x2],
                                face_mesh_landmarks,
                                self.config.face_mesh_options,
                                self.config,
                            )
                        if self.config.eye_color['enable']:
                            eye_color_name = detect_eye_color(face_roi, face_mesh_landmarks)
                            if eye_color_name:
                                eye_color_text_color_bgr = self.config.eye_color_text_color[::-1]
                                cv2.putText(
                                    annotated_frame,
                                    f"Eye Color: {eye_color_name}",
                                    (x1, y2 + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    eye_color_text_color_bgr,
                                    2,
                                )

                detection_info = {
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': cls,
                    'name': name,
                    'similarity': similarity,
                    'blink': blink if self.config.blink['enable'] else None,
                    'face_mesh': bool(face_mesh_landmarks) if self.config.face_mesh_options['enable'] else False,
                    'hands_detected': bool(hand_landmarks_list),
                    'hand_count': len(hand_landmarks_list) if hand_landmarks_list else 0,
                    'eye_color': eye_color_name if self.config.eye_color['enable'] and eye_color_name else None,
                }
                results.append(detection_info)

            return annotated_frame, results

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, []

    def is_real_face(self, face_roi: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            result = lap_var > self.config.anti_spoof['lap_thresh']
            logger.debug(f"Anti-spoofing result: {result} (Laplacian Variance: {lap_var})")
            return result
        except Exception as e:
            logger.error(f"Anti-spoof check failed: {str(e)}")
            return False

    def recognize_face(self, embedding: np.ndarray, recognition_threshold: float) -> Tuple[str, float]:
        try:
            best_match = "Unknown"
            best_similarity = 0.0
            for label, embeddings in self.db.embeddings.items():
                for db_emb in embeddings:
                    similarity = self.cosine_similarity(embedding, db_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = label
            if best_similarity < recognition_threshold:
                best_match = "Unknown"
            logger.debug(f"Recognized as {best_match} with similarity {best_similarity:.2f}")
            return best_match, best_similarity
        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            return "Unknown", 0.0

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def main():
        import argparse
        import cv2

        parser = argparse.ArgumentParser(description="Face Recognition Pipeline")
        parser.add_argument('--mode', choices=['webcam', 'image', 'enroll', 'share'], required=False, default='gui',
                            help="Mode to run the pipeline: 'webcam', 'image', 'enroll', 'share', or 'gui'")
        parser.add_argument('--input', help="Input file path for image mode or share mode")
        parser.add_argument('--label', help="Label for enrollment")
        args = parser.parse_args()

        try:
            config = PipelineConfig.load(config_path=args.input if args.mode == 'share' else 'face_pipeline/config.pkl')
            pipeline = FacePipeline(config)
            pipeline.initialize()

            if args.mode == 'webcam':
                cap = cv2.VideoCapture(0)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed, detections = pipeline.process_frame(frame)
                    cv2.imshow('Face Recognition', processed)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

            elif args.mode == 'image':
                image = cv2.imread(args.input)
                if image is None:
                    logger.error(f"Failed to load image from {args.input}")
                    print(f"Error: Failed to load image from {args.input}")
                    sys.exit(1)
                processed, detections = pipeline.process_frame(image)
                cv2.imwrite("processed.jpg", processed)
                print("Saved processed image to processed.jpg")

            elif args.mode == 'enroll':
                if not args.label or not args.input:
                    logger.error("Enrollment mode requires --label and --input arguments.")
                    print("Error: Enrollment mode requires --label and --input arguments.")
                    sys.exit(1)
                image = cv2.imread(args.input)
                if image is None:
                    logger.error(f"Failed to load image from {args.input}")
                    print(f"Error: Failed to load image from {args.input}")
                    sys.exit(1)
                detections = pipeline.detector.detect(image, pipeline.config.detection_conf_thres)
                if not detections:
                    logger.error("No faces detected in the enrollment image.")
                    print("Error: No faces detected in the enrollment image.")
                    sys.exit(1)
                for x1, y1, x2, y2, conf, cls in detections:
                    face_roi = image[y1:y2, x1:x2]
                    if face_roi.size == 0:
                        continue
                    emb = pipeline.facenet.get_embedding(face_roi)
                    if emb is not None:
                        pipeline.db.add_embedding(args.label, emb)
                pipeline.db.save()
                print(f"Enrolled {args.label} successfully")

            elif args.mode == 'share':
                # Implement sharing functionality here
                from .utilities import upload_pipeline
                if not args.input:
                    logger.error("Share mode requires --input argument (pipeline file path).")
                    print("Error: Share mode requires --input argument (pipeline file path).")
                    sys.exit(1)
                success = upload_pipeline(args.input, destination_url='https://yourserver.com/upload')
                if success:
                    print("Pipeline shared successfully!")
                else:
                    print("Failed to share pipeline.")

            else:
                # Run Streamlit GUI if no mode is specified
                from .gui import run_streamlit_gui
                run_streamlit_gui()

        except Exception as e:
            logger.critical(f"Main execution failed: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")
            sys.exit(1)
