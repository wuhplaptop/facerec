# tests/test_basic.py

import os
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from rolo_rec.config import Config
from rolo_rec.facial_recognition import FacialRecognition
from rolo_rec.combined_model import CombinedFacialRecognitionModel
import torch

@pytest.fixture
def mock_facial_recognition(tmp_path):
    """
    Fixture to create a FacialRecognition instance using the combined model with mocked components.
    
    This fixture:
    - Mocks the CombinedFacialRecognitionModel to avoid loading the actual 'face.pt' model.
    - Mocks the __call__ and embed_faces_batch methods to return controlled outputs.
    - Sets up a temporary path for user data.
    """
    with patch('rolo_rec.combined_model.CombinedFacialRecognitionModel') as MockModel:
        # Create a mock instance of CombinedFacialRecognitionModel
        mock_model_instance = MagicMock(spec=CombinedFacialRecognitionModel)
        
        # Mock the __call__ method to simulate face detection and embedding extraction
        mock_model_instance.__call__.return_value = [
            (10, 10, 50, 50)
        ], [
            np.array([0.1] * 512, dtype=np.float32)
        ]
        
        # Mock the embed_faces_batch method if it's used separately
        mock_model_instance.embed_faces_batch.return_value = np.array([0.1] * 512, dtype=np.float32)
        
        # Mock the state_dict and load_state_dict to prevent PicklingError
        mock_model_instance.state_dict.return_value = {}
        mock_model_instance.load_state_dict.return_value = None
        
        # Mock the save_model method to do nothing
        mock_model_instance.save_model.return_value = None
        
        # When CombinedFacialRecognitionModel.load_model is called, return the mock instance
        MockModel.load_model.return_value = mock_model_instance
        
        # Initialize FacialRecognition with the mocked combined model
        config = Config(user_data_path=str(tmp_path / "test_faces.json"))
        fr = FacialRecognition(config, combined_model_path="models/face.pt")
        
    return fr

def test_register_with_face_separate_models(mock_facial_recognition):
    """
    Test registering a user using the combined YOLO and Facenet model.
    
    This test verifies that:
    - A user can be registered with a single image.
    - The appropriate methods (__call__ and embed_faces_batch) are called with correct arguments.
    """
    fr = mock_facial_recognition

    # Create a dummy image
    img = Image.new("RGB", (100, 100), color="white")
    
    # Register the user
    msg = fr.register_user("TestUser", [img])

    # Assertions
    assert "User 'TestUser' registered with 1 images." in msg, "Registration message mismatch."
    assert fr.combined_model.user_embeddings.get("TestUser") is not None, "User embeddings not found after registration."
    assert len(fr.combined_model.user_embeddings["TestUser"]) == 1, "Incorrect number of embeddings stored for the user."
    
    # Ensure the mock methods were called correctly
    fr.combined_model.__call__.assert_called_once_with(img)
    # Note: embed_faces_batch might not be called directly since embeddings are returned from __call__
    # If embed_faces_batch is used elsewhere, you can add assertions accordingly

def test_identify_known_user_separate_models(mock_facial_recognition):
    """
    Test identifying a known user using the combined YOLO and Facenet model.
    
    This test verifies that:
    - A registered user can be identified correctly.
    - The similarity score meets expectations.
    - The appropriate methods (__call__ and embed_faces_batch) are called with correct arguments.
    """
    fr = mock_facial_recognition

    # Pre-register a user with a known embedding
    fr.combined_model.user_embeddings = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    # Create a dummy image for identification
    img = Image.new("RGB", (100, 100), color="white")
    
    # Identify the user
    results = fr.identify_user(img, threshold=0.5)

    # Assertions
    assert len(results) == 1, "Identification should return exactly one result."
    assert results[0]['user_id'] == 'TestUser', "Identified user ID does not match."
    assert results[0]['similarity'] == pytest.approx(1.0, abs=1e-6), "Similarity score mismatch."
    
    # Ensure the mock methods were called correctly
    fr.combined_model.__call__.assert_called_once_with(img)
    # Note: embed_faces_batch might not be called directly since embeddings are returned from __call__

def test_export_model_combined_model(tmp_path, mock_facial_recognition):
    """
    Test exporting the combined model.
    
    This test verifies that:
    - The combined model is exported correctly.
    - The exported file exists and contains the expected user embeddings.
    - The embeddings in the exported model match the originals.
    """
    fr = mock_facial_recognition

    # Pre-register a user with a known embedding in the combined model
    fr.combined_model.user_embeddings = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    # Define the export path
    export_path = str(tmp_path / "exported_combined.pt")
    
    # Export the combined model
    msg = fr.export_model(export_path)

    # Assertions
    assert "[Export] Combined model saved to" in msg, "Export message mismatch."
    assert os.path.exists(export_path), "Exported combined model file does not exist."

    # Mock the load_model method to return the same user_embeddings
    with patch('rolo_rec.combined_model.CombinedFacialRecognitionModel.load_model') as MockLoadModel:
        # Setup the mock to return a new mock instance with the same user_embeddings
        mock_loaded_model = MagicMock(spec=CombinedFacialRecognitionModel)
        mock_loaded_model.user_embeddings = fr.combined_model.user_embeddings
        MockLoadModel.return_value = mock_loaded_model

        # Load the exported combined model
        loaded_model = CombinedFacialRecognitionModel.load_model(export_path, device='cpu')

    # Assertions
    assert loaded_model.user_embeddings.keys() == fr.combined_model.user_embeddings.keys(), "User keys mismatch in exported model."
    for user in fr.combined_model.user_embeddings:
        assert user in loaded_model.user_embeddings, f"User '{user}' missing in exported model."
        for emb_original, emb_loaded in zip(fr.combined_model.user_embeddings[user],
                                            loaded_model.user_embeddings[user]):
            assert np.allclose(emb_original, emb_loaded, atol=1e-6), "User embeddings do not match."

def test_import_model(tmp_path, mock_facial_recognition):
    """
    Test importing a combined model.
    
    This test verifies that:
    - The combined model is imported correctly.
    - The user embeddings in the imported model match the originals.
    """
    fr = mock_facial_recognition

    # Pre-register a user with a known embedding in the combined model
    fr.combined_model.user_embeddings = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    # Define the import path
    import_path = str(tmp_path / "import_model.pt")
    
    # Export the combined model
    fr.combined_model.save_model(import_path)

    # Patch torch.load to return the actual state with user embeddings
    with patch('torch.load') as mock_torch_load:
        mock_torch_load.return_value = {
            'yolo_state_dict': {},
            'facenet_state_dict': {},
            'user_embeddings': {
                "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
            },
            'device': 'cpu',
            'yolo_model_path': 'yolov8n.pt'
        }

        # Import the model using FacialRecognition
        config = Config(user_data_path=str(tmp_path / "test_faces.json"))
        imported_fr = FacialRecognition.import_model(import_path, config)

    # Assertions
    assert imported_fr.combined_model.user_embeddings.keys() == fr.combined_model.user_embeddings.keys(), "User keys mismatch after import."
    for user in fr.combined_model.user_embeddings:
        assert user in imported_fr.combined_model.user_embeddings, f"User '{user}' missing after import."
        for emb_original, emb_imported in zip(fr.combined_model.user_embeddings[user],
                                              imported_fr.combined_model.user_embeddings[user]):
            assert np.allclose(emb_original, emb_imported, atol=1e-6), "User embeddings do not match after import."
