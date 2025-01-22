import pytest
from unittest.mock import MagicMock, patch
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.combined_model import CombinedFacialRecognitionModel
from myfacerec.data_store import UserDataStore
from PIL import Image
import numpy as np

# Mock configuration object
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "myfacerec/models/face.pt"
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None  # Assuming no alignment function for simplicity

@pytest.fixture
def mock_config():
    """Mock configuration fixture."""
    return MockConfig()

@pytest.fixture
def mock_combined_model():
    """
    Mock CombinedFacialRecognitionModel instance using MagicMock with a spec.
    The __call__ method is redirected to the mocked forward method.
    """
    model = MagicMock(spec=CombinedFacialRecognitionModel)

    # Mock the forward method so that model(...) returns something
    model.forward.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]
    model.__call__.side_effect = model.forward

    # Some tests do: mock_combined_model.detect_and_embed.return_value = ...
    # So define detect_and_embed on the mock:
    model.detect_and_embed.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]

    # Mock the user_embeddings attribute
    model.user_embeddings = {}

    # Mock sub‐attributes
    model.yolo = MagicMock()
    model.facenet = MagicMock()

    # Mock state_dict for yolo and facenet
    model.yolo.model.state_dict.return_value = {'yolo_layer': 'yolo_weights'}
    model.facenet.model.state_dict.return_value = {'facenet_layer': 'facenet_weights'}

    # Ensure save_model can handle (self, save_path)
    def _mock_save_model(save_path):
        # No actual file I/O
        pass
    model.save_model.side_effect = _mock_save_model

    return model

@pytest.fixture
def mock_data_store(mock_combined_model):
    """Mock UserDataStore instance linked to model.user_embeddings."""
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    """
    Mocked FacialRecognition that uses our CombinedFacialRecognitionModel
    and a mocked data store.
    """
    fr = FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )
    return fr


# ---------------------------------------------------------------------------
# Below are the original 4 tests that previously failed due to mocking issues
# ---------------------------------------------------------------------------

def test_register_user(mock_facial_recognition, mock_combined_model):
    """Test registering a new user."""
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]  # Mock image objects

    # We want detect_and_embed to return a face with an embedding
    mock_combined_model.detect_and_embed.return_value = [
        ((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))
    ]

    message = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in message.lower()
    assert user_id in mock_facial_recognition.user_data

def test_identify_user_known(mock_facial_recognition, mock_combined_model):
    """Test identifying a known user."""
    user_id = "known_user"
    # Put a known embedding in user_data
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # detect_and_embed returns something close to that known embedding
    mock_combined_model.detect_and_embed.return_value = [
        ((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))
    ]

    # Pretend we have an image
    img_mock = MagicMock(spec=Image.Image)
    results = mock_facial_recognition.identify_user(img_mock)
    assert len(results) == 1
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition, mock_combined_model):
    """Test identifying an unknown user."""
    user_id = "known_user"
    # Put a known embedding in user_data
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    # detect_and_embed returns something dissimilar
    mock_combined_model.detect_and_embed.return_value = [
        ((10, 10, 100, 100), np.array([0.4, 0.5, 0.6]))
    ]

    img_mock = MagicMock(spec=Image.Image)
    results = mock_facial_recognition.identify_user(img_mock)
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, mock_combined_model, tmp_path):
    """Test exporting the facial recognition model."""
    export_path = tmp_path / "exported_model.pt"

    # Create some user data
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Setup YOLO/facenet mock for state_dict
    yolo_state_dict = {'yolo_layer': 'yolo_weights'}
    facenet_state_dict = {'facenet_layer': 'facenet_weights'}
    mock_combined_model.yolo.model.state_dict.return_value = yolo_state_dict
    mock_combined_model.facenet.model.state_dict.return_value = facenet_state_dict

    # Attempt the export
    mock_facial_recognition.export_combined_model(str(export_path))

    # Because we mocked save_model, we just confirm it was called
    mock_combined_model.save_model.assert_called_once_with(str(export_path))
