# tests/test_basic.py

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# -----------------------------------------------------------------
# 1) Patch YOLO in myfacerec.combined_model so it won't do file IO
# -----------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def patch_model_yolo():
    """
    Patch myfacerec.combined_model.YOLO with a MagicMock that won't load .pt files.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_class:
        dummy_yolo = MagicMock()
        dummy_yolo.to.return_value = dummy_yolo
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}
        # By default, dummy_yolo(image) => no detections
        dummy_yolo.__call__ = MagicMock(return_value=[])
        yolo_class.return_value = dummy_yolo
        yield

# Now import your code after the patch is in place
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# -----------------------------------------------------------------
# 2) Mock Config
# -----------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# -----------------------------------------------------------------
# 3) Real CombinedFacialRecognitionModel (no real .pt loaded thanks to patch)
# -----------------------------------------------------------------
@pytest.fixture
def mock_combined_model(mock_config):
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold
    )
    # Keep real .yolo and .facenet submodules
    return model

# -----------------------------------------------------------------
# 4) Data store fixture
# -----------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    ds = MagicMock(spec=UserDataStore)
    ds.load_user_data.return_value = mock_combined_model.user_embeddings
    ds.save_user_data.return_value = None
    return ds

# -----------------------------------------------------------------
# 5) FacialRecognition fixture
# -----------------------------------------------------------------
@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    return FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )

# -----------------------------------------------------------------
# 6) Tests
# -----------------------------------------------------------------

def test_register_user(mock_facial_recognition):
    """
    Mocks a single YOLO detection with conf=0.99, class=0, bounding box.
    Mocks facenet.forward => embedding [0.1,0.2,0.3].
    Expects "registered" message, not "No valid face embeddings found".
    """
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # YOLO => one bounding box with real Tensors
    detection_mock = MagicMock()
    box_mock = MagicMock()

    # conf => shape [1], cls => shape [1]
    box_mock.conf = torch.tensor([0.99])
    box_mock.cls  = torch.tensor([0])
    # xyxy => shape [1,4]
    box_mock.xyxy = torch.tensor([[10, 10, 100, 100]], dtype=torch.float)

    detection_mock.boxes = [box_mock]
    # Override self.model.yolo(...) => returns this detection
    mock_facial_recognition.model.yolo.__call__ = MagicMock(
        return_value=[detection_mock]
    )

    # Facenet => real tensor embedding
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    """
    If embedding matches known_user data => 1 result, user_id=known_user.
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf = torch.tensor([0.99])
    box_mock.cls  = torch.tensor([0])
    box_mock.xyxy = torch.tensor([[10, 10, 100, 100]], dtype=torch.float)
    detection_mock.boxes = [box_mock]

    mock_facial_recognition.model.yolo.__call__ = MagicMock(return_value=[detection_mock])
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition):
    """
    If embedding is dissimilar => "Unknown".
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf = torch.tensor([0.99])
    box_mock.cls  = torch.tensor([0])
    box_mock.xyxy = torch.tensor([[10, 10, 100, 100]], dtype=torch.float)
    detection_mock.boxes = [box_mock]

    mock_facial_recognition.model.yolo.__call__ = MagicMock(return_value=[detection_mock])
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.4, 0.5, 0.6]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, tmp_path):
    """
    Overriding yolo.model.state_dict() and facenet.state_dict() to real dict
    avoids pickling Mocks. Then exporting should succeed.
    """
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Overwrite with real dict returning lambdas
    mock_facial_recognition.model.yolo.model.state_dict = lambda: {}
    mock_facial_recognition.model.facenet.state_dict = lambda: {}

    mock_facial_recognition.export_combined_model(str(export_path))
    # If we get here, no pickling error
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
