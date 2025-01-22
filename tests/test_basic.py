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
# 1) Patching YOLO so it won't try to load any .pt file
# ---------------------------------------------------------------------
@pytest.fixture
def patch_yolo(monkeypatch):
    """
    Patch the YOLO class so it doesn't load any actual weights file.
    We'll override the entire constructor plus any relevant methods.
    """

    class DummyYOLO:
        def __init__(self, model=None, task=None, verbose=True):
            # do nothing
            self.model = MagicMock()
            self.model.state_dict = MagicMock(return_value={})
            self.predict = MagicMock(return_value=[])
            # if you call self(...) it might return results, but we will mock it in tests

        def to(self, device):
            return self

        def __call__(self, image, *args, **kwargs):
            # By default, return no detections. We'll override in each test
            return []

    monkeypatch.setattr("ultralytics.models.yolo.model.YOLO", DummyYOLO)

# ---------------------------------------------------------------------
# 2) A simple config
# ---------------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"  # won't actually load
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# ---------------------------------------------------------------------
# 3) Create a real CombinedFacialRecognitionModel
#    but rely on the patched YOLO constructor
# ---------------------------------------------------------------------
@pytest.fixture
def mock_combined_model(patch_yolo):
    """
    This fixture uses the patched YOLO, so it won't fail on missing .pt file.
    Then we can still override model.yolo & model.facenet with Mocks.
    """
    model = CombinedFacialRecognitionModel(
        yolo_model_path="fake_yolo_model.pt",  # won't load due to patch
        device="cpu",
        conf_threshold=0.5
    )
    # Replace YOLO & Facenet with mocks
    model.yolo = MagicMock()
    model.facenet = MagicMock()

    # user_embeddings is an empty dict by default
    model.user_embeddings = {}
    return model

# ---------------------------------------------------------------------
# 4) Data store fixture
# ---------------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

# ---------------------------------------------------------------------
# 5) FacialRecognition fixture
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
# 6) Tests that actually mock YOLO bounding boxes & Facenet embedding
# ---------------------------------------------------------------------

def test_register_user(mock_facial_recognition):
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # YOLO returns a single bounding box
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Facenet returns [0.1,0.2,0.3]
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

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
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, tmp_path):
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Just ensure we can call export without crashing
    mock_facial_recognition.export_combined_model(str(export_path))
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
