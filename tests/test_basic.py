# tests/test_basic.py

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from PIL import Image

from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# ---------------------------------------------------------------------
# Mock config class (basic placeholder)
# ---------------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "myfacerec/models/face.pt"
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# ---------------------------------------------------------------------
# Fixture: real CombinedFacialRecognitionModel but YOLO & facenet are mocked
# ---------------------------------------------------------------------
@pytest.fixture
def mock_combined_model():
    """
    Return a real CombinedFacialRecognitionModel instance so that
    forward(...) logic runs. Then we replace model.yolo & model.facenet
    with mocks, so we can force bounding boxes and embeddings without
    calling the real YOLO or facenet code.
    """
    model = CombinedFacialRecognitionModel(
        yolo_model_path="fake_yolo_model.pt",  # won't be used
        device="cpu",
        conf_threshold=0.5
    )
    # Replace YOLO and facenet with MagicMock
    model.yolo = MagicMock()
    model.facenet = MagicMock()

    # Link user_embeddings to an empty dict
    model.user_embeddings = {}
    return model

# ---------------------------------------------------------------------
# Fixture: mock data store
# ---------------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    """
    Fake user data store. Ties load_user_data to model.user_embeddings,
    so we read/write the same dictionary.
    """
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

# ---------------------------------------------------------------------
# Fixture: mock FacialRecognition that uses our mock model & data store
# ---------------------------------------------------------------------
@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    fr = FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )
    return fr

# ---------------------------------------------------------------------
# TEST 1: test_register_user
# ---------------------------------------------------------------------
def test_register_user(mock_facial_recognition):
    """
    Succeeds if exactly 1 face is found => "registered".
    Otherwise: "No valid face embeddings found..."
    """
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    #
    # Mock YOLO: Return 1 bounding box with conf>0.5, class=0
    #
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99  # confidence
    box_mock.cls.item.return_value = 0      # class=0 => face
    box_mock.xyxy = [
        torch.tensor([10, 10, 100, 100], dtype=torch.float)
    ]
    detection_mock.boxes = [box_mock]
    # So self.yolo(img) => [detection_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    #
    # Mock facenet: Return embedding [0.1, 0.2, 0.3] for that face
    #
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    #
    # Now do the real registration flow
    #
    msg = mock_facial_recognition.register_user(user_id, images)

    #
    # Verify we see "registered" in the message
    #
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    # We expect exactly 1 embedding for that user now
    assert len(mock_facial_recognition.user_data[user_id]) == 1

# ---------------------------------------------------------------------
# TEST 2: test_identify_user_known
# ---------------------------------------------------------------------
def test_identify_user_known(mock_facial_recognition):
    """
    If the embedding is similar to known_user's existing embedding,
    we expect 1 face recognized as 'known_user'.
    """
    user_id = "known_user"
    # Insert a known embedding
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # Mock YOLO detection => 1 bounding box
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Mock facenet => returns [0.1, 0.2, 0.3]
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == user_id

# ---------------------------------------------------------------------
# TEST 3: test_identify_user_unknown
# ---------------------------------------------------------------------
def test_identify_user_unknown(mock_facial_recognition):
    """
    If the face embedding doesn't match any user data, we get "Unknown".
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # 1 bounding box from YOLO
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Return a dissimilar embedding => "Unknown"
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.4, 0.5, 0.6]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == "Unknown"

# ---------------------------------------------------------------------
# TEST 4: test_export_model
# ---------------------------------------------------------------------
def test_export_model(mock_facial_recognition, tmp_path):
    """
    Simple check that export_combined_model calls .save_model
    and no errors occur.
    """
    export_path = tmp_path / "exported_model.pt"

    # Insert some user data
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # We don't need YOLO/facenet here, just ensure no crash
    msg = mock_facial_recognition.export_combined_model(str(export_path))

    # .save_model is the final step
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
