# myfacerec/plugins/sample_plugin.py

from ..detectors import FaceDetector

class SampleDetector(FaceDetector):
    def __init__(self):
        # Initialize your custom detector here
        pass

    def detect_faces(self, image):
        # Implement custom detection logic
        # For demonstration, return an empty list
        return []
