import pytest
from unittest.mock import MagicMock, patch
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.config import Config
from myfacerec.combined_model import CombinedFacialRecognitionModel
from PIL import Image
import numpy as np
import torch

@pytest.fixture
def mock_config(tmp_path):
    """Mock configuration object."""
    return Config(
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
        user_data_path=str(tmp_path / "user_data.json"),
        yolo_model_path=None,
        default_model_url="https://example.com/face.pt",
        conf_threshold=0.5,
    )

@pytest.fixture
def mock_combined_model(tmp_path):
    """Mock CombinedFacialRecognitionModel."""
    with patch("myfacerec.combined_model.CombinedFacialRecognitionModel") as MockModel:
        # Create a mock instance
        mock_model_instance = MagicMock(spec=CombinedFacialRecognitionModel)
        
        # Mock methods
        mock_model_instance.forward.return_value = [
            ((10, 10, 50, 50), np.array([0.1] * 512, dtype=np.float32))
        ]
        mock_model_instance.user_embeddings = {}
        mock_model_instance.load_model.return_value = mock_model_instance
        mock_model_instance.save_model.return_value = None
        
        # Save mock to a path
        mock_file = tmp_path / "combined_model.pt"
        mock_file.touch()  # Create an empty mock file for testing
        
        return MockModel

@pytest.fixture
def mock_facial_recognition(mock_config, tmp_path, mock_combined_model):
    """Mock FacialRecognition instance."""
    combined_model_path = str(tmp_path / "combined_model.pt")
    return FacialRecognition(config=mock_config, combined_model_path=combined_model_path)

def test_register_with_face_combined_model(mock_facial_recognition):
    """Test registering a new face with the combined model."""
    face_image = Image.new("RGB", (224, 224))  # Dummy image
    user_id = "user_123"

    result = mock_facial_recognition.register_user(user_id=user_id, images=[face_image])
    assert "registered" in result, f"Unexpected result: {result}"

def test_identify_known_user_combined_model(mock_facial_recognition):
    """Test identifying a known user with the combined model."""
    face_image = Image.new("RGB", (224, 224))  # Dummy image

    # Mock user embeddings
    mock_facial_recognition.combined_model.user_embeddings = {
        "user_123": [np.array([0.1] * 512, dtype=np.float32)]
    }

    results = mock_facial_recognition.identify_user(image=face_image)
    assert len(results) > 0, "No users identified!"
    assert results[0]["user_id"] == "user_123", f"Unexpected user ID: {results[0]['user_id']}"

def test_export_combined_model(mock_facial_recognition, tmp_path):
    """Test exporting the combined model."""
    export_path = tmp_path / "exported_model.pt"
    result = mock_facial_recognition.export_model(str(export_path))
    assert "saved" in result, f"Unexpected result: {result}"
    assert export_path.exists(), "Exported model file does not exist!"

def test_import_combined_model(mock_config, tmp_path, mock_combined_model):
    """Test importing a combined model."""
    import_path = tmp_path / "imported_model.pt"
    import_path.touch()  # Create a dummy file

    with patch("myfacerec.combined_model.CombinedFacialRecognitionModel.load_model") as mock_load:
        mock_load.return_value = mock_combined_model
        fr = FacialRecognition.import_model(str(import_path), config=mock_config)
        assert fr.combined_model is not None, "Failed to import combined model!"
