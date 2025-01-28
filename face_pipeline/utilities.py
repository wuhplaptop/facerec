# face_pipeline/utilities.py

import cv2
import numpy as np
from typing import Tuple
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for blink detection.
    
    Parameters:
        eye_landmarks (np.ndarray): Array of eye landmark coordinates.
    
    Returns:
        float: EAR value.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks (p2-p6 and p3-p5)
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Compute the euclidean distance between the horizontal eye landmark (p1-p4)
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(face_roi: np.ndarray, threshold: float = 0.2, consecutive_frames: int = 3) -> Tuple[bool, float, float, np.ndarray, np.ndarray]:
    """
    Detects if a person is blinking based on eye aspect ratio (EAR).
    
    Parameters:
        face_roi (np.ndarray): The region of interest containing the face.
        threshold (float): The EAR threshold to detect a blink.
        consecutive_frames (int): Number of consecutive frames the EAR must be below the threshold.
    
    Returns:
        Tuple containing:
            - blink (bool): Whether a blink was detected.
            - left_ear (float): Left eye aspect ratio.
            - right_ear (float): Right eye aspect ratio.
            - left_eye (np.ndarray): Coordinates of the left eye.
            - right_eye (np.ndarray): Coordinates of the right eye.
    """
    blink = False
    left_ear = 0.0
    right_ear = 0.0
    left_eye = np.array([])
    right_eye = np.array([])
    
    try:
        rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Define landmark indices for left and right eyes (using Mediapipe's face mesh)
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [263, 387, 385, 362, 380, 373]
            
            left_eye_coords = []
            right_eye_coords = []
            
            for idx in left_eye_indices:
                x = face_landmarks.landmark[idx].x * face_roi.shape[1]
                y = face_landmarks.landmark[idx].y * face_roi.shape[0]
                left_eye_coords.append((x, y))
            
            for idx in right_eye_indices:
                x = face_landmarks.landmark[idx].x * face_roi.shape[1]
                y = face_landmarks.landmark[idx].y * face_roi.shape[0]
                right_eye_coords.append((x, y))
            
            left_eye = np.array(left_eye_coords)
            right_eye = np.array(right_eye_coords)
            
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            ear = (left_ear + right_ear) / 2.0
            if ear < threshold:
                blink = True
    except Exception as e:
        # Handle exceptions (e.g., no face detected)
        pass
    
    return blink, left_ear, right_ear, left_eye, right_eye
