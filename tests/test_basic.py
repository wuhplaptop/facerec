import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from myfacerec.combined_model import CombinedFacialRecognitionModel
from myfacerec.facial_recognition import FacialRecognition

@pytest.fixture
def mock_facial_recognition(tmp_path):
    """
    Fixture to create a FacialRecognition instance using the combined model with mocked components.

    This fixture:
    - Mocks the CombinedFacialRecognitionModel to avoid loading the actual 'face.pt' model.
    - Mocks the forward and embed_faces_batch methods to return controlled outputs.
    - Sets up a temporary path for user data.
    """
    with patch('myfacerec.combined_model.CombinedFacialRecognitionModel') as MockModel:
        # Create a mock instance of CombinedFacialRecognitionModel
        mock_model_instance = MagicMock(spec=['forward', 'embed_faces_batch'])

        # Mock the forward method to simulate face detection and embedding extraction
        mock_model_instance.forward.return_value = [
            (10, 10, 50, 50)
        ], [
            np.array([0.1] * 512, dtype=np.float32)
        ]

        # Explicitly mock the embed_faces_batch method
        mock_model_instance.embed_faces_batch.return_value = [
            np.array([0.1] * 512, dtype=np.float32)
        ]

        # MockModel returns our mock_model_instance
        MockModel.return_value = mock_model_instance

        # Initialize the FacialRecognition instance with the mock model and temp path
        fr_instance = FacialRecognition(combined_model=mock_model_instance, data_dir=tmp_path)

        yield fr_instance

def test_register_with_face_separate_models(mock_facial_recognition):
    """
    Test registering a new user with a face using the separate models.
    """
    user_id = "test_user"
    image_path = "path/to/test_image.jpg"

    result = mock_facial_recognition.register_user(user_id, image_path)

    assert result is True, "User registration failed."

def test_identify_known_user_separate_models(mock_facial_recognition):
    """
    Test identifying a known user with a face using the separate models.
    """
    user_id = "test_user"
    image_path = "path/to/test_image.jpg"

    mock_facial_recognition.register_user(user_id, image_path)

    identified_user = mock_facial_recognition.identify_user(image_path)

    assert identified_user == user_id, "Failed to identify the known user."

def test_export_model_combined_model(mock_facial_recognition):
    """
    Test exporting a combined facial recognition model.
    """
    export_path = "path/to/exported_model.pt"

    result = mock_facial_recognition.export_model(export_path)

    assert result is True, "Model export failed."

def test_import_model(mock_facial_recognition):
    """
    Test importing a combined facial recognition model.
    """
    import_path = "path/to/imported_model.pt"

    result = mock_facial_recognition.import_model(import_path)

    assert result is True, "Model import failed."
