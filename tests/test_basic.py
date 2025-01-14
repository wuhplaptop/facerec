import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from myfacerec.facial_recognition import FacialRecognition

@pytest.fixture
def mock_facial_recognition(tmp_path):
    """
    Fixture to create a FacialRecognition instance using a mock model.
    This fixture:
    - Mocks the model to avoid loading the actual 'face.pt' model.
    - Mocks the `forward` and `embed_faces_batch` methods to return controlled outputs.
    - Sets up a temporary path for user data.
    """
    with patch('myfacerec.combined_model.CombinedFacialRecognitionModel') as MockModel:
        # Create a mock instance of the model
        mock_model_instance = MagicMock()
        
        # Mock the forward method to simulate face detection and embedding extraction
        mock_model_instance.forward.return_value = [
            (10, 10, 50, 50)
        ], [
            np.array([0.1] * 512, dtype=np.float32)
        ]
        
        # Mock the embed_faces_batch method
        mock_model_instance.embed_faces_batch.return_value = [
            np.array([0.1] * 512, dtype=np.float32)
        ]
        
        # MockModel returns the mock_model_instance
        MockModel.return_value = mock_model_instance
        
        # Initialize the FacialRecognition instance
        fr_instance = FacialRecognition(model=mock_model_instance, data_dir=tmp_path)
        
        return fr_instance

def test_register_with_face_separate_models(mock_facial_recognition):
    """
    Test registering a new face using separate models.
    """
    fr_instance = mock_facial_recognition
    face_image = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy face image
    user_id = "user_123"
    
    result = fr_instance.register_face(user_id=user_id, face_image=face_image)
    assert result, "Face registration failed!"

def test_identify_known_user_separate_models(mock_facial_recognition):
    """
    Test identifying a known user using separate models.
    """
    fr_instance = mock_facial_recognition
    face_image = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy face image
    
    user_id = fr_instance.identify_face(face_image)
    assert user_id is not None, "Failed to identify a known user!"

def test_export_model_combined_model(mock_facial_recognition, tmp_path):
    """
    Test exporting the combined model.
    """
    fr_instance = mock_facial_recognition
    export_path = tmp_path / "exported_model.pt"
    
    fr_instance.export_model(str(export_path))
    assert export_path.exists(), "Model export failed!"

def test_import_model(mock_facial_recognition, tmp_path):
    """
    Test importing a model.
    """
    fr_instance = mock_facial_recognition
    import_path = tmp_path / "imported_model.pt"
    import_path.touch()  # Create a dummy file to simulate an existing model
    
    fr_instance.import_model(str(import_path))
    assert fr_instance.model is not None, "Model import failed!"
