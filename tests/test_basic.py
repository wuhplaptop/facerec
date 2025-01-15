# tests/test_basic.py

import pytest
from unittest.mock import patch, MagicMock
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.combined_model import CombinedFacialRecognitionModel
from myfacerec.data_store import UserDataStore
from pathlib import Path
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
    """Mock CombinedFacialRecognitionModel instance."""
    model = MagicMock(spec=CombinedFacialRecognitionModel)
    # Mock the __call__ method to return detections
    # Each detection is a tuple of (bounding_box, embedding)
    model.__call__.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]
    # Mock the save_model method
    model.save_model.return_value = None
    # Mock the load_model class method
    CombinedFacialRecognitionModel.load_model = MagicMock(return_value=model)
    return model

@pytest.fixture
def mock_data_store():
    """Mock UserDataStore instance."""
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = {}
    data_store.save_user_data.return_value = None
    return data_store

@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    """Mock FacialRecognition instance with mock combined model and data store."""
    with patch("myfacerec.facial_recognition.JSONUserDataStore", return_value=mock_data_store):
        fr = FacialRecognition(
            config=mock_config,
            data_store=mock_data_store
        )
        fr.model = mock_combined_model
    return fr

# Test cases
def test_facial_recognition_initialization(mock_config, mock_combined_model, mock_data_store):
    """Test initialization of the FacialRecognition class."""
    # Act
    fr = FacialRecognition(
        config=mock_config,
        data_store=mock_data_store
    )
    fr.model = mock_combined_model

    # Assert
    assert fr.config == mock_config
    assert fr.model == mock_combined_model
    assert fr.data_store == mock_data_store
    assert fr.user_data == mock_combined_model.user_embeddings
    mock_data_store.load_user_data.assert_called_once()

def test_register_user(mock_facial_recognition):
    """Test registering a new user."""
    # Arrange
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]  # Mock image objects

    # Mock model __call__ to return one face with embedding
    mock_facial_recognition.model.__call__.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]

    # Act
    message = mock_facial_recognition.register_user(user_id, images)

    # Assert
    mock_facial_recognition.model.__call__.assert_called_once_with(images[0])
    mock_facial_recognition.data_store.save_user_data.assert_called_once_with(mock_facial_recognition.user_data)
    assert message == f"User '{user_id}' registered with 1 embedding(s)."
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1
    assert np.array_equal(mock_facial_recognition.user_data[user_id][0], np.array([0.1, 0.2, 0.3]))

def test_identify_user_known(mock_facial_recognition):
    """Test identifying a known user."""
    # Arrange
    user_id = "known_user"
    mock_facial_recognition.user_data = {
        user_id: [np.array([0.1, 0.2, 0.3])]
    }

    # Mock model __call__ to return embeddings similar to known user
    mock_facial_recognition.model.__call__.return_value = [((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))]

    # Mock cosine_similarity to return perfect similarity
    with patch("myfacerec.facial_recognition.cosine_similarity", return_value=np.array([[1.0]])):
        results = mock_facial_recognition.identify_user(Image.new('RGB', (100, 100)))

    # Assert
    expected_result = [{'face_id': 1, 'user_id': user_id, 'similarity': 1.0}]
    assert results == expected_result

def test_identify_user_unknown(mock_facial_recognition):
    """Test identifying an unknown user."""
    # Arrange
    user_id = "known_user"
    mock_facial_recognition.user_data = {
        user_id: [np.array([0.1, 0.2, 0.3])]
    }

    # Mock model __call__ to return embeddings dissimilar to known user
    mock_facial_recognition.model.__call__.return_value = [((10, 10, 100, 100), np.array([0.4, 0.5, 0.6]))]

    # Mock cosine_similarity to return low similarity
    with patch("myfacerec.facial_recognition.cosine_similarity", return_value=np.array([[0.5]])):
  
