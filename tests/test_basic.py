# tests/test_basic.py

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from PIL import Image

# ---------------------------------------------------------------------
# 1) Patch our local YOLO import so it never tries to load real files
# ---------------------------------------------------------------------
from unittest.mock import patch

@pytest.fixture(scope="session", autouse=True)
def patch_model_yolo():
    """
    Globally patch `myfacerec.combined_model.YOLO` so it never tries
    to load an actual .pt file.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_mock:
        # Create a dummy YOLO instance
        dummy_yolo = MagicMock()
        dummy_yolo.to.return_value = dummy_yolo
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}
        dummy_yolo.__call__.return_value = []

        # So any time CombinedFacialRecognitionModel does `YOLO(...)`,
        # it gets `dummy_yolo`.
        yolo_mock.return_value = dummy_yolo

        yield

# ---------------------------------------------------------------------
# 2) Now import your classes after the patch
# ---------------------------------------------------------------------
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# ---------------------------------------------------------------------
# 3) Mock config
# ---------------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"  # won't load due to patch
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# ---------------------------------------------------------------------
# 4) mock_combined_model uses the real constructor, but YOLO is patched
# ---------------------------------------------------------------------
@pytest.fixture
def mock_combined_model(mock_config):
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold
    )
    # We'll mock model.yolo & model.facenet for bounding boxes/embeddings
    model.yolo = MagicMock()
    model.facenet = MagicMock()
    model.user_embeddings = {}
    return model

# ---------------------------------------------------------------------
# 5) Data store
# ---------------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

# ---------------------------------------------------------------------
# 6) FacialRecognition
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
# 7) Tests
# ---------------------------------------------------------------------
def test_register_user(mock_facial_recognition):
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # YOLO => 1 bounding box
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Facenet => [0.1,0.2,0.3]
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # YOLO => 1 bounding box
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Facenet => matching embedding
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # YOLO => 1 bounding box
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Facenet => dissimilar embedding => "Unknown"
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.4, 0.5, 0.6]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, tmp_path):
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]
    mock_facial_recognition.export_combined_model(str(export_path))
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
