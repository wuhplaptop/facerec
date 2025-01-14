# tests/test_basic.py

import pytest
from unittest.mock import patch, MagicMock
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.detectors import FaceDetector
from myfacerec.embedders import FaceEmbedder
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
        self.yolo_model_path = "models/face.pt"
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None  # Assuming no alignment function for simplicity


# Fixtures for tests
@pytest.fixture
def mock_config():
    """Mock configuration fixture."""
    return MockConfig()


@pytest.fixture
def mock_detector():
    """Mock FaceDetector instance."""
    detector = MagicMock(spec=FaceDetector)
    detector.detect_faces.return_value = [(10, 10, 100, 100)]  # Mocked bounding boxes
    detector.model = MagicMock()
    detector.model.state_dict.return_value = {'mock_key': 'mock_value'}
    return detector


@pytest.fixture
def mock_embedder():
    """Mock FaceEmbedder instance."""
    embedder = MagicMock(spec=FaceEmbedder)
    embedder.embed_faces_batch.return_value = np.array([[0.1, 0.2, 0.3]])  # Mocked embeddings
    embedder.model = MagicMock()
    embedder.model.state_dict.return_value = {'mock_embedding_key': 'mock_embedding_value'}
    return embedder


@pytest.fixture
def mock_data_store():
    """Mock UserDataStore instance."""
    data_store = MagicMock(spec=UserDataStore)
    data_store.load_user_data.return_value = {}
    data_store.save_user_data.return_value = None
    return data_store


@pytest.fixture
def mock_facial_recognition(mock_config, mock_detector, mock_embedder, mock_data_store):
    """Mock FacialRecognition instance with mock detector, embedder, and data store."""
    with patch("myfacerec.facial_recognition.JSONUserDataStore", return_value=mock_data_store):
        fr = FacialRecognition(
            config=mock_config,
            detector=mock_detector,
            embedder=mock_embedder,
            data_store=mock_data_store
        )
    return fr


# Test cases
def test_facial_recognition_initialization(mock_config, mock_detector, mock_embedder, mock_data_store):
    """Test initialization of the FacialRecognition class."""
    # Act
    fr = FacialRecognition(
        config=mock_config,
        detector=mock_detector,
        embedder=mock_embedder,
        data_store=mock_data_store
    )

    # Assert
    assert fr.config == mock_config
    assert fr.detector == mock_detector
    assert fr.embedder == mock_embedder
    assert fr.data_store == mock_data_store


def test_register_user(mock_facial_recognition):
    """Test registering a new user."""
    # Arrange
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]  # Mock image objects

    # Mock detect_faces to return a single face
    mock_facial_recognition.detector.detect_faces.return_value = [(10, 10, 100, 100)]

    # Mock embed_faces_batch to return an embedding
    mock_facial_recognition.embedder.embed_faces_batch.return_value = np.array([[0.1, 0.2, 0.3]])

    # Act
    message = mock_facial_recognition.register_user(user_id, images)

    # Assert
    mock_facial_recognition.detector.detect_faces.assert_called_once_with(images[0])
    mock_facial_recognition.embedder.embed_faces_batch.assert_called_once_with(images[0], [(10, 10, 100, 100)])
    mock_facial_recognition.data_store.save_user_data.assert_called_once()
    assert message == f"User '{user_id}' registered with 1 valid face embedding(s)."
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1
    assert mock_facial_recognition.user_data[user_id][0].tolist() == [0.1, 0.2, 0.3]


def test_identify_user_known(mock_facial_recognition):
    """Test identifying a known user."""
    # Arrange
    user_id = "known_user"
    mock_facial_recognition.user_data = {
        user_id: [np.array([0.1, 0.2, 0.3])]
    }
    embeddings = np.array([[0.1, 0.2, 0.3]])  # Embedding similar to known user

    # Mock cosine_similarity to return perfect similarity
    with patch("myfacerec.facial_recognition.cosine_similarity", return_value=np.array([[1.0]])):
        results = mock_facial_recognition.identify_user(embeddings, threshold=0.6)

    # Assert
    expected_result = [{'user_id': user_id, 'similarity': 1.0}]
    assert results == expected_result


def test_identify_user_unknown(mock_facial_recognition):
    """Test identifying an unknown user."""
    # Arrange
    user_id = "known_user"
    mock_facial_recognition.user_data = {
        user_id: [np.array([0.1, 0.2, 0.3])]
    }
    embeddings = np.array([[0.4, 0.5, 0.6]])  # Embedding dissimilar to known user

    # Mock cosine_similarity to return low similarity
    with patch("myfacerec.facial_recognition.cosine_similarity", return_value=np.array([[0.5]])):
        results = mock_facial_recognition.identify_user(embeddings, threshold=0.6)

    # Assert
    expected_result = [{'user_id': 'Unknown', 'similarity': 0.5}]
    assert results == expected_result


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
    mock_facial_recognition.detector.model.state_dict.return_value = yolo_state_dict
    mock_facial_recognition.embedder.model.state_dict.return_value = facenet_state_dict

    # Expected state to be saved (without calling state_dict())
    expected_state = {
        'yolo_state_dict': yolo_state_dict,
        'facenet_state_dict': facenet_state_dict,
        'user_embeddings': {
            "user1": [np.array([0.1, 0.2, 0.3])]
        },
        'device': mock_facial_recognition.config.device
    }

    # Mock torch.save to verify it's called correctly
    with patch("torch.save") as mock_torch_save:
        # Act
        mock_facial_recognition.export_model(str(export_path))

        # Assert
        # Ensure state_dict() is called exactly once for each model
        mock_facial_recognition.detector.model.state_dict.assert_called_once()
        mock_facial_recognition.embedder.model.state_dict.assert_called_once()

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

        # Assert device
        assert saved_state['device'] == mock_facial_recognition.config.device, "Device does not match."

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
