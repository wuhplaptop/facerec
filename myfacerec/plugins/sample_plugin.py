# myfacerec/plugins/sample_plugin.py

from ..detectors import FaceDetector
from PIL import Image

class SampleDetector(FaceDetector):
    def __init__(self):
        # Initialize your custom detector here
        pass

    def detect_faces(self, image: Image.Image):
        # Implement custom detection logic
        # For demonstration, return an empty list
        return []
