import os
import pytest
import numpy as np
from PIL import Image
from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.plugins.sample_plugin import SampleDetector

def test_register_with_face(tmp_path):
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path)
    fr = FacialRecognition(config)

    def mock_detect_faces(image):
        return [(10, 10, 50, 50)]

    def mock_embed_faces_batch(image, boxes):
        # Return a (1, 512) NumPy array
        return np.array([[0.1]*512], dtype=np.float32)

    fr.detect_faces = mock_detect_faces
    fr.embed_faces_batch = mock_embed_faces_batch

    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("TestUser", [img])

    assert "User 'TestUser' registered with 1 images." in msg
    assert fr.user_data.get("TestUser") is not None
    assert len(fr.user_data["TestUser"]) == 1

def test_identify_known_user(tmp_path):
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path)
    fr = FacialRecognition(config)

    # The "TestUser" embedding
    fr.user_data = {
        "TestUser": [np.array([0.1]*512, dtype=np.float32)]
    }

    def mock_detect_faces(image):
        return [(10, 10, 50, 50)]

    def mock_embed_faces_batch(image, boxes):
        # Return a (1, 512) NumPy array
        return np.array([[0.1]*512], dtype=np.float32)

    fr.detect_faces = mock_detect_faces
    fr.embed_faces_batch = mock_embed_faces_batch

    img = Image.new("RGB", (100, 100), color="white")
    results = fr.identify_user(img, threshold=0.5)

    assert len(results) == 1
    assert results[0]['user_id'] == 'TestUser'
    # Change from exact equality to approximate
    assert results[0]['similarity'] == pytest.approx(1.0, abs=1e-6)
