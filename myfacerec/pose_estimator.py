# myfacerec/pose_estimator.py

import cv2
import numpy as np

class HeadPoseEstimator:
    """
    Estimates head pose (yaw, pitch, roll) given face landmarks.
    For demonstration, we define a rough 3D model of certain face landmarks.
    You must provide real 2D landmark coordinates that correspond to these 3D points.
    """
    def __init__(self):
        # Rough 3D facial landmark coordinates (could be improved or replaced).
        self.model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye (left corner)
            (225.0, 170.0, -135.0),     # Right eye (right corner)
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ], dtype=np.float32)

    def estimate_pose(self, image, face_landmarks_2d):
        """
        Returns (yaw, pitch, roll) in degrees, or None if it fails.

        Args:
            image (np.ndarray): BGR or RGB image (H, W, C).
            face_landmarks_2d (np.ndarray): shape (6, 2) or more,
                2D coordinates for the same keypoints that match self.model_points.
        """
        if face_landmarks_2d.shape[0] < len(self.model_points):
            return None

        # Image shape
        height, width = image.shape[:2]
        focal_length = width  # a rough guess for focal length
        center = (width / 2, height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # No lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve for head pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points.astype(np.float32),
            face_landmarks_2d.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        # Convert rotation vector to rotation matrix -> euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # For yaw, pitch, roll: see typical formulas
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])   # pitch
        y = np.arctan2(-rotation_matrix[2, 0], sy)                    # yaw
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])   # roll

        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        return (yaw, pitch, roll)
