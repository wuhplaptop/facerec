import pytest
from unittest.mock import MagicMock, patch
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel
from PIL import Image
import numpy as np

# Mock config
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

@pytest.fixture
def mock_combined_model():
    """
    MagicMock without 'spec=CombinedFacialRecognitionModel'
    so we can mock __call__.side_effect without triggering
    'method object has no attribute side_effect' errors.
    """
    # 1) Create a generic MagicMock (no spec)
    model = MagicMock()

    # 2) Let .forward(...) return something
    model.forward.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]

    # 3) Make model(...) call model.forward(...)
    #    This is how PyTorch usually works, but we can do it manually here.
    model.__call__.side_effect = model.forward

    # 4) If your tests do mock_combined_model.detect_and_embed.return_value = ...
    #    define a default return:
    model.detect_and_embed.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]

    # 5) Mock user_embeddings, sub-attributes, etc.
    model.user_embeddings = {}
    model.yolo = MagicMock()
    model.facenet = MagicMock()

    model.yolo.model.state_dict.return_value = {'yolo_layer': 'yolo_weights'}
    model.facenet.model.state_dict.return_value = {'facenet_layer': 'facenet_weights'}

    # 6) Mock save_model so it won’t do real file I/O
    def _mock_save_model(save_path):
        pass
    model.save_model.side_effect = _mock_save_model

    return model

@pytest.fixture
def mock_data_store(mock_combined_model):
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    """Mock FacialRecognition that uses the above mock model & data store."""
    fr = FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )
    return fr

# ------------------------------------------------------------------------
# Example tests that used to fail due to mocking issues:
# ------------------------------------------------------------------------

def test_register_user(mock_facial_recognition, mock_combined_model):
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]
    mock_combined_model.detect_and_embed.return_value = [
        ((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))
    ]
    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower()
    assert user_id in mock_facial_recognition.user_data

def test_identify_user_known(mock_facial_recognition, mock_combined_model):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]
    mock_combined_model.detect_and_embed.return_value = [
        ((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))
    ]
    img_mock = MagicMock(spec=Image.Image)
    results = mock_facial_recognition.identify_user(img_mock)
    assert len(results) == 1
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition, mock_combined_model):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]
    # Return a dissimilar embedding
    mock_combined_model.detect_and_embed.return_value = [
        ((10, 10, 100, 100), np.array([0.4, 0.5, 0.6]))
    ]
    img_mock = MagicMock(spec=Image.Image)
    results = mock_facial_recognition.identify_user(img_mock)
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, mock_combined_model, tmp_path):
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]
    # Mock YOLO & facenet state
    yolo_dict = {"yolo_layer": "yolo_weights"}
    face_dict = {"facenet_layer": "facenet_weights"}
    mock_combined_model.yolo.model.state_dict.return_value = yolo_dict
    mock_combined_model.facenet.model.state_dict.return_value = face_dict
    # Export
    mock_facial_recognition.export_combined_model(str(export_path))
    mock_combined_model.save_model.assert_called_once_with(str(export_path))
