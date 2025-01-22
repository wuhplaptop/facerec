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

# Fixtures for tests
@pytest.fixture
def mock_config():
    """Mock configuration fixture."""
    return MockConfig()

@pytest.fixture
def mock_combined_model():
    """
    Mock CombinedFacialRecognitionModel instance using MagicMock with spec.
    The __call__ method is redirected to the mocked forward method.
    """
    # The 'spec=' ensures method signatures match the real class.
    model = MagicMock(spec=CombinedFacialRecognitionModel)  # <-- CHANGED (still the same but keep it noted)

    # Mock the forward method so that model(...) returns something
    model.forward.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]
    model.__call__.side_effect = model.forward

    # The tests do: mock_combined_model.detect_and_embed.return_value = ...
    # So define detect_and_embed on the mock:
    model.detect_and_embed.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]  # <-- ADDED

    # Mock the user_embeddings attribute
    model.user_embeddings = {}

    # Mock the yolo and facenet attributes with their own mocks
    model.yolo = MagicMock()
    model.facenet = MagicMock()

    # Mock the state_dict method for yolo and facenet models
    model.yolo.model.state_dict.return_value = {'yolo_layer': 'yolo_weights'}
    model.facenet.model.state_dict.return_value = {'facenet_layer': 'facenet_weights'}

    # Mock the save_model method; it must accept two parameters:
    # (self, save_path). By default, Python treats the first param as 'self'.
    def _mock_save_model(save_path):  # <-- CHANGED
        # No-op to prevent file IO
        pass

    model.save_model.side_effect = _mock_save_model  # <-- CHANGED

    return model

@pytest.fixture
def mock_data_store(mock_combined_model):
    """Mock UserDataStore instance."""
    data_store = MagicMock(spec=UserDataStore)
    # Link user_data to model.user_embeddings
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    """Mock FacialRecognition instance with mock combined model and data store."""
    # Initialize FacialRecognition with the mocked model and data store
    fr = FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model  # Injecting the mock model
    )
    return fr


# ...the rest of your tests remain unchanged...
