# myfacerec/pose_estimator.py

import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        # You can store your 3D model landmarks or camera parameters here
        # For example, a rough 3D model of 6-7 essential landmarks 
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # You could store a default camera matrix if you know the focal length / principal point
        self.camera_matrix = None  # will set per image if needed

    def estimate_pose(self, image, face_landmarks_2d):
        """
        image: BGR or RGB image
        face_landmarks_2d: list/array of (x, y) for the same landmarks that map to self.model_points
        Returns yaw, pitch, roll in degrees
        """
        if len(face_landmarks_2d) < len(self.model_points):
            return None  # not enough landmarks

        size = image.shape
        focal_length = size[1]  # approximate
        center = (size[1] / 2, size[0] / 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # We assume no lens distortion
        dist_coeffs = np.zeros((4, 1))  

        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            face_landmarks_2d,
            self.camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix -> euler angles (yaw, pitch, roll)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # [yaw, pitch, roll] might need an explicit function or a standard approach
        # For example:
        sy = np.sqrt(rotation_matrix[0, 0]*rotation_matrix[0, 0] +  rotation_matrix[1, 0]*rotation_matrix[1, 0])
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        return yaw, pitch, roll
