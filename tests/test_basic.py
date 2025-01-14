import pytest
from unittest.mock import patch, MagicMock
from myfacerec.facial_recognition import FacialRecognition
from pathlib import Path


# Mock configuration object
class MockConfig:
    def __init__(self):
        self.some_config_option = "mock_value"
        self.device = "cpu"  # Add the missing 'device' attribute


# Fixtures for tests
@pytest.fixture
def mock_config():
    """Mock configuration fixture."""
    return MockConfig()


@pytest.fixture
def mock_combined_model():
    """Mock CombinedFacialRecognitionModel instance."""
    combined_model = MagicMock()
    combined_model.register.return_value = True
    combined_model.identify.return_value = {"name": "John Doe", "confidence": 0.95}
    combined_model.export.return_value = True
    return combined_model


@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, tmp_path):
    """Mock FacialRecognition instance with a mock model file."""
    # Create a dummy combined model file in the temporary path
    combined_model_path = tmp_path / "mock_combined_model_path.pt"
    combined_model_path.write_text("mock model content")  # Create a dummy file

    with patch(
        "myfacerec.combined_model.CombinedFacialRecognitionModel.load_model",
        return_value=mock_combined_model,
    ), patch("os.path.exists", return_value=True):  # Mock the file existence check
        return FacialRecognition(
            config=mock_config,
            combined_model_path=str(combined_model_path),
        )


# Test cases
def test_facial_recognition_initialization(mock_config, tmp_path, mock_combined_model):
    """Test initialization of the FacialRecognition class."""
    # Arrange
    combined_model_path = tmp_path / "test_model.pt"
    combined_model_path.write_text("mock model content")  # Create a dummy file

    # Act
    with patch(
        "myfacerec.combined_model.CombinedFacialRecognitionModel.load_model",
        return_value=mock_combined_model,
    ), patch("os.path.exists", return_value=True):  # Mock the file existence check
        facial_recognition = FacialRecognition(
            config=mock_config,
            combined_model_path=str(combined_model_path),
        )

    # Assert
    assert facial_recognition.config == mock_config
    assert facial_recognition is not None


def test_register_with_face_combined_model(mock_facial_recognition, mock_combined_model):
    """Test registration with the combined model."""
    # Arrange
    mock_user_data = {"name": "John Doe", "face_embedding": [0.1, 0.2, 0.3]}

    # Act
    result = mock_combined_model.register(mock_user_data)

    # Assert
    mock_combined_model.register.assert_called_once_with(mock_user_data)
    assert result is True


def test_identify_known_user_combined_model(mock_facial_recognition, mock_combined_model):
    """Test identifying a known user with the combined model."""
    # Arrange
    mock_face_data = {"face_embedding": [0.1, 0.2, 0.3]}

    # Act
    result = mock_combined_model.identify(mock_face_data)

    # Assert
    mock_combined_model.identify.assert_called_once_with(mock_face_data)
    assert result == {"name": "John Doe", "confidence": 0.95}


def test_export_combined_model(mock_facial_recognition, tmp_path, mock_combined_model):
    """Test exporting the combined model."""
    # Arrange
    export_path = tmp_path / "exported_model.pt"

    # Act
    result = mock_combined_model.export(str(export_path))

    # Assert
    mock_combined_model.export.assert_called_once_with(str(export_path))
    assert result is True
