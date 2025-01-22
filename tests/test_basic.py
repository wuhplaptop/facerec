import pytest
from unittest.mock import MagicMock
from PIL import Image
import numpy as np

from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# A simple mock config
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
    Creates a generic MagicMock to represent the CombinedFacialRecognitionModel.
    We explicitly mock model.__call__ because your production code calls self.model(img).
    """
    model = MagicMock()

    # By default, calling model(...) will do nothing;
    # we override its .return_value in each test as needed.
    model.__call__ = MagicMock()

    # We can also mock save_model, user_embeddings, yolo, facenet, etc.
    model.save_model = MagicMock()
    model.user_embeddings = {}
    model.yolo = MagicMock()
    model.facenet = MagicMock()
    model.yolo.model.state_dict.return_value = {"yolo_layer": "yolo_weights"}
    model.facenet.model.state_dict.return_value = {"facenet_layer": "facenet_weights"}

    return model

@pytest.fixture
def mock_data_store(mock_combined_model):
    # Tie the data store to the mock model’s user_embeddings
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    """
    Creates a FacialRecognition instance that uses:
      - the mock config
      - the mock data store
      - the mock combined model
    """
    fr = FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )
    return fr


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test_register_user(mock_facial_recognition, mock_combined_model):
    """
    Should succeed with "registered" if exactly 1 face is detected in the image.
    """
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # Force the model to detect exactly 1 face
    mock_combined_model.__call__.return_value = [
        ((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))
    ]

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition, mock_combined_model):
    """
    Should return one result matching 'known_user' if embedding is similar.
    """
    user_id = "known_user"
    # Insert a known embedding
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # Force the model to detect a face with a matching embedding
    mock_combined_model.__call__.return_value = [
        ((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))
    ]

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face result, got {len(results)}"
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition, mock_combined_model):
    """
    Should return "Unknown" if the detected face embedding doesn't match any user.
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # Return a dissimilar embedding => "Unknown"
    mock_combined_model.__call__.return_value = [
        ((10, 10, 100, 100), np.array([0.4, 0.5, 0.6]))
    ]

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face result, got {len(results)}"
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, mock_combined_model, tmp_path):
    """
    Test exporting the facial recognition model. We check that save_model is called.
    """
    export_path = tmp_path / "exported_model.pt"

    # Insert some user embeddings so there's something to export
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Act
    mock_facial_recognition.export_combined_model(str(export_path))

    # Assert
    mock_combined_model.save_model.assert_called_once_with(str(export_path))
