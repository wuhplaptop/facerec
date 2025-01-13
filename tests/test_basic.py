# tests/test_basic.py

import os
import pytest
from PIL import Image
from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition

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
