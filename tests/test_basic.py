import pytest
import torch
import os
from unittest.mock import MagicMock
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.combined_model import CombinedFacialRecognitionModel

# Mock configuration class for testing
@pytest.fixture
def mock_config():
    class MockConfig:
        device = "cpu"
    return MockConfig()

# Mock CombinedFacialRecognitionModel
@pytest.fixture
def mock_combined_model(mocker):
    mock_model = mocker.patch('myfacerec.combined_model.CombinedFacialRecognitionModel', autospec=True)
    mock_model.load_model.return_value = MagicMock()
    return mock_model

# Create a mock facial recognition fixture
@pytest.fixture
def mock_facial_recognition(mock_config, tmp_path, mock_combined_model):
    """Mock FacialRecognition instance with a valid combined model."""
    combined_model_path = str(tmp_path / "combined_model.pt")

    # Ensure the combined model file exists
    if not os.path.exists(combined_model_path):
        torch.save({"mock_state_dict": {}}, combined_model_path)

    return FacialRecognition(config=mock_config, combined_model_path=combined_model_path)

def test_register_with_face_combined_model(mock_facial_recognition):
    """Test registering a user with the combined model."""
    mock_facial_recognition.register_user("test_user", "path/to/test/image.jpg")
    # Add assertions based on expected behavior

def test_identify_known_user_combined_model(mock_facial_recognition):
    """Test identifying a known user with the combined model."""
    result = mock_facial_recognition.identify_user("path/to/test/image.jpg")
    assert result == "expected_user", "User identification failed."

def test_export_combined_model(mock_facial_recognition, tmp_path):
    """Test exporting the combined model."""
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.export_model(str(export_path))
    assert os.path.exists(export_path), "Exported model file does not exist."

def test_facial_recognition_initialization(mock_facial_recognition):
    """Test initializing the FacialRecognition class."""
    assert isinstance(mock_facial_recognition, FacialRecognition), "Failed to initialize FacialRecognition."
