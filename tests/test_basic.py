# tests/test_basic.py

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
    model = MagicMock(spec=CombinedFacialRecognitionModel)
    
    # Mock the forward method to return detections and embeddings
    # Each detection is a tuple of (bounding_box, embedding)
    model.forward.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]
    
    # Redirect __call__ to forward
    model.__call__.side_effect = model.forward
    
    # Mock the user_embeddings attribute
    model.user_embeddings = {}
    
    # Mock the yolo and facenet attributes with their own mocks
    model.yolo = MagicMock()
    model.facenet = MagicMock()
    
    # Mock the state_dict method for yolo and facenet models
    model.yolo.model.state_dict.return_value = {'yolo_layer': 'yolo_weights'}
    model.facenet.model.state_dict.return_value = {'facenet_layer': 'facenet_weights'}
    
    # Mock the save_model method to prevent actual file operations
    model.save_model.return_value = None
    
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
