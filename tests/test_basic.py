# tests/test_basic.py

import os
import pytest
from PIL import Image
from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.plugins.sample_plugin import SampleDetector

def test_register_no_faces(tmp_path):
    # Use a temp data file so we don't overwrite real user data
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path, conf_threshold=0.99)  # very high => no detection
    fr = FacialRecognition(config)

    # Create a blank image (likely no faces)
    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("NoFaceUser", [img])

    assert "No valid single-face images" in msg
    assert not fr.user_data.get("NoFaceUser")

def test_register_with_face(tmp_path):
    # Use a temp data file
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path)
    fr = FacialRecognition(config)

    # Mock image with one face by assuming detection returns one box
    def mock_detect_faces(image):
        return [(10, 10, 50, 50)]

    def mock_embed_faces_batch(image, boxes):
        return [[0.1] * 512]  # Mock embedding

    fr.detect_faces = mock_detect_faces
    fr.embed_faces_batch = mock_embed_faces_batch

    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("TestUser", [img])

    assert "User 'TestUser' registered with 1 images." in msg
    assert fr.user_data.get("TestUser") is not None
    assert len(fr.user_data["TestUser"]) == 1

def test_identify_unknown(tmp_path):
    # Setup with no users
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path)
    fr = FacialRecognition(config)

    img = Image.new("RGB", (100, 100), color="white")
    results = fr.identify_user(img)

    assert len(results) == 0

def test_identify_known_user(tmp_path):
    # Setup with one user
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path)
    fr = FacialRecognition(config)

    # Mock embedding for a user
    fr.user_data = {
        "TestUser": [ [0.1] * 512 ]
    }

    # Mock detection and embedding
    def mock_detect_faces(image):
        return [(10, 10, 50, 50)]

    def mock_embed_faces_batch(image, boxes):
        return [ [0.1] * 512 ]  # Same embedding as TestUser

    fr.detect_faces = mock_detect_faces
    fr.embed_faces_batch = mock_embed_faces_batch

    img = Image.new("RGB", (100, 100), color="white")
    results = fr.identify_user(img, threshold=0.5)

    assert len(results) == 1
    assert results[0]['user_id'] == 'TestUser'
    assert results[0]['similarity'] == 1.0

def test_plugin_loading():
    config = Config()
    fr = FacialRecognition(config, detector=SampleDetector())
    assert isinstance(fr.detector, SampleDetector)
