import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.combined_model import CombinedFacialRecognitionModel


class MockConfig:
    def __init__(self):
        self.device = "cpu"
        self.some_setting = "mock_setting"
        self.yolo_model_path = None
        self.default_model_url = "https://example.com/default_model.pt"
        self.conf_threshold = 0.5
        self.cache_dir = "./cache"
        self.user_data_path = "./user_data.json"
        self.alignment_fn = None
        self.detector_plugin = None
        self.embedder_plugin = None


@pytest.fixture
def mock_config():
    """Provide a mock configuration."""
    return MockConfig()


@pytest.fixture
def mock_combined_model():
    """Provide a mock combined facial recognition model."""
    return MagicMock(spec=CombinedFacialRecognitionModel)


@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model):
    """Mock FacialRecognition instance."""
    with patch(
        "myfacerec.combined_model.CombinedFacialRecognitionModel.load_model",
        return_value=mock_combined_model,
    ):
        facial_recognition = FacialRecognition(
            config=mock_config,
            combined_model_path="mock_combined_model_path.pt"
        )
        facial_recognition.combined_model = mock_combined_model
        return facial_recognition


def test_register_with_face_combined_model(mock_facial_recognition):
    """Test registering a face with the combined model."""
    # Arrange
    user_id = "user123"
    face_image = MagicMock()

    # Act
    result = mock_facial_recognition.register_user(user_id, [face_image])

    # Assert
    assert "User" in result and user_id in result
    mock_facial_recognition.combined_model.register_user.assert_called_once()


def test_identify_known_user_combined_model(mock_facial_recognition):
    """Test identifying a known user with the combined model."""
    # Arrange
    face_image = MagicMock()
    mock_facial_recognition.combined_model.identify_user.return_value = [
        {"user_id": "user123", "similarity": 0.9}
    ]

    # Act
    result = mock_facial_recognition.identify_user(face_image)

    # Assert
    assert isinstance(result, list)
    assert result[0]["user_id"] == "user123"
    mock_facial_recognition.combined_model.identify_user.assert_called_once_with(face_image)


def test_export_combined_model(mock_facial_recognition, tmp_path):
    """Test exporting the combined model."""
    # Arrange
    export_path = tmp_path / "exported_model.pt"

    # Act
    result = mock_facial_recognition.export_model(str(export_path))

    # Assert
    assert "saved to" in result
    mock_facial_recognition.combined_model.save_model.assert_called_once_with(str(export_path))


def test_facial_recognition_initialization(mock_config, tmp_path, mock_combined_model):
    """Test initialization of the FacialRecognition class."""
    # Arrange
    combined_model_path = tmp_path / "test_model.pt"

    # Act
    with patch(
        "myfacerec.combined_model.CombinedFacialRecognitionModel.load_model",
        return_value=mock_combined_model,
    ):
        facial_recognition = FacialRecognition(
            config=mock_config,
            combined_model_path=str(combined_model_path)
        )
        facial_recognition.combined_model = mock_combined_model

    # Assert
    assert facial_recognition.config == mock_config
    assert facial_recognition.combined_model_path == str(combined_model_path)
    assert facial_recognition.combined_model == mock_combined_model
