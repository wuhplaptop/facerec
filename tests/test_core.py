# tests/test_core.py

import unittest
import numpy as np
import cv2
from face_pipeline.config import PipelineConfig
from face_pipeline.core import FacePipeline

class TestFacePipeline(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig()
        self.pipeline = FacePipeline(self.config)
        self.pipeline.initialize()

    def test_process_frame_no_face(self):
        # Create a blank image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed, detections = self.pipeline.process_frame(frame)
        self.assertIsNotNone(processed)
        self.assertEqual(len(detections), 0)

    def test_process_frame_with_face(self):
        # Load a sample image with a face
        image_path = 'tests/sample_face.jpg'
        frame = cv2.imread(image_path)
        if frame is None:
            self.skipTest("Sample face image not found.")
        processed, detections = self.pipeline.process_frame(frame)
        self.assertIsNotNone(processed)
        self.assertGreater(len(detections), 0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
