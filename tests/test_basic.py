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
    # Create a MagicMock with the spec of CombinedFacialRecognitionModel
    model = MagicMock(spec=CombinedFacialRecognitionModel)
    
    # Mock the __call__ method to return detections and embeddings
    # Each detection is a tuple of (bounding_box, embedding)
    model.__call__ = MagicMock(return_value=[((10, 10, 100, 100), np.array([0.1, 0.2, 0.3]))])
    
    # Mock the user_embeddings attribute
    model.user_embeddings = {}
    
    # Mock the yolo and facenet attributes with their own mocks
    model.yolo = MagicMock()
    model.facenet = MagicMock()
    
    # Mock the state_dict method for yolo and facenet models
    model.yolo.model = MagicMock()
    model.facenet.model = MagicMock()
    model.yolo.model.state_dict = MagicMock(return_value={'yolo_layer': 'yolo_weights'})
    model.facenet.model.state_dict = MagicMock(return_value={'facenet_layer': 'facenet_weights'})
    
    # Mock the save_model method to prevent actual file operations
    model.save_model = MagicMock(return_value=None)
    
    # Mock the load_model class method to return this mocked model
    with patch.object(CombinedFacialRecognitionModel, 'load_model', return_value=model):
        yield model

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
def test_facial_recognition_initialization(mock_facial_recognition, mock_config, mock_combined_model, mock_data_store):
    """Test initialization of the FacialRecognition class."""
    # Act is already done via fixture
    fr = mock_facial_recognition

    # Assert
    assert fr.config == mock_config, "Config does not match."
    assert fr.model == mock_combined_model, "Model does not match."
    assert fr.data_store == mock_data_store, "Data store does not match."
    assert fr.user_data == mock_combined_model.user_embeddings, "User data does not match."
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
    assert message == f"User '{user_id}' registered with 1 embedding(s).", "Registration message mismatch."
    assert user_id in mock_facial_recognition.user_data, "User ID not in user data."
    assert len(mock_facial_recognition.user_data[user_id]) == 1, "Incorrect number of embeddings for user."
    assert np.array_equal(mock_facial_recognition.user_data[user_id][0], np.array([0.1, 0.2, 0.3])), "Embedding does not match."

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
    assert results == expected_result, f"Expected {expected_result}, but got {results}."

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
        results = mock_facial_recognition.identify_user(Image.new('RGB', (100, 100)))

    # Assert
    expected_result = [{'face_id': 1, 'user_id': 'Unknown', 'similarity': 0.5}]
    assert results == expected_result, f"Expected {expected_result}, but got {results}."

def test_export_model(mock_facial_recognition, tmp_path):
    """Test exporting the facial recognition model."""
    # Arrange
    export_path = tmp_path / "exported_model.pt"

    # Populate user_data to ensure it's included in the export
    mock_facial_recognition.user_data = {
        "user1": [np.array([0.1, 0.2, 0.3])]
    }

    # Define mock return values for state_dict
    yolo_state_dict = {'yolo_layer': 'yolo_weights'}
    facenet_state_dict = {'facenet_layer': 'facenet_weights'}

    # Assign the predefined return values to the mocks
    mock_facial_recognition.model.yolo.model.state_dict.return_value = yolo_state_dict
    mock_facial_recognition.model.facenet.model.state_dict.return_value = facenet_state_dict

    # Expected state to be saved
    expected_state = {
        'yolo_state_dict': yolo_state_dict,
        'facenet_state_dict': facenet_state_dict,
        'user_embeddings': {
            "user1": [np.array([0.1, 0.2, 0.3])]
        },
        'config': {
            'yolo_model_path': mock_facial_recognition.config.yolo_model_path,
            'conf_threshold': mock_facial_recognition.config.conf_threshold,
            'device': mock_facial_recognition.config.device
        }
    }

    # Mock torch.save to verify it's called correctly
    with patch("torch.save") as mock_torch_save:
        # Act
        mock_facial_recognition.export_combined_model(str(export_path))

        # Assert
        # Ensure state_dict() is called exactly once for each model
        mock_facial_recognition.model.yolo.model.state_dict.assert_called_once()
        mock_facial_recognition.model.facenet.model.state_dict.assert_called_once()

        # Ensure torch.save is called once
        mock_torch_save.assert_called_once()

        # Retrieve the actual call arguments
        args, kwargs = mock_torch_save.call_args
        saved_state = args[0]
        saved_export_path = args[1]

        # Assert that the export path is correct
        assert saved_export_path == str(export_path), "Export path does not match."

        # Assert yolo_state_dict
        assert saved_state['yolo_state_dict'] == yolo_state_dict, "YOLO state_dict does not match."

        # Assert facenet_state_dict
        assert saved_state['facenet_state_dict'] == facenet_state_dict, "Facenet state_dict does not match."

        # Assert config
        assert saved_state['config'] == {
            'yolo_model_path': mock_facial_recognition.config.yolo_model_path,
            'conf_threshold': mock_facial_recognition.config.conf_threshold,
            'device': mock_facial_recognition.config.device
        }, "Config does not match."

        # Assert user_embeddings
        assert "user1" in saved_state['user_embeddings'], "User1 not found in user_embeddings."
        assert len(saved_state['user_embeddings']['user1']) == 1, "Incorrect number of embeddings for user1."
        np.testing.assert_array_almost_equal(
            saved_state['user_embeddings']['user1'][0],
            mock_facial_recognition.user_data['user1'][0],
            decimal=5,
            err_msg="User embedding does not match."
        )

        # Ensure save_user_data is NOT called
        mock_facial_recognition.data_store.save_user_data.assert_not_called()
