# tests/test_basic.py

import os
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.plugins.sample_plugin import SampleDetector
from myfacerec.combined_model import CombinedFacialRecognitionModel
import torch

@pytest.fixture
def mock_facial_recognition(tmp_path):
    """
    Fixture to create a FacialRecognition instance with mocked detect_faces and embed_faces_batch methods.
    """
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path)

    # Patch the YOLO class to prevent actual model loading
    with patch('myfacerec.facial_recognition.YOLO') as MockYOLO:
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model.state_dict.return_value = {}
        MockYOLO.return_value = mock_yolo_instance

        fr = FacialRecognition(config)

        # Mock detect_faces to return a single bounding box and a single embedding
        fr.detect_faces = MagicMock(return_value=([(10, 10, 50, 50)], [np.array([0.1] * 512, dtype=np.float32)]))

        # Mock embed_faces_batch to return a single embedding
        fr.embed_faces_batch = MagicMock(return_value=np.array([[0.1]*512], dtype=np.float32))

    return fr

@pytest.fixture
def mock_combined_model(tmp_path):
    """
    Fixture to create a FacialRecognition instance using a CombinedFacialRecognitionModel with mocked embeddings.
    """
    with patch('myfacerec.combined_model.YOLO') as MockYOLO:
        mock_yolo = MagicMock()
        mock_yolo.model.state_dict.return_value = {}
        MockYOLO.return_value = mock_yolo

        # Initialize the combined model
        combined_model = CombinedFacialRecognitionModel(yolo_model_path="yolov8n.pt", device="cpu")
        combined_model.user_embeddings = {}  # Start with no users

        # Mock the forward method to return predefined boxes and embeddings
        combined_model.forward = MagicMock(return_value=[((10, 10, 50, 50), np.array([0.1] * 512, dtype=np.float32))])

        # Save the mocked model to a temporary path
        combined_model_path = str(tmp_path / "combined_model.pt")
        combined_model.save_model(combined_model_path)

        # Patch torch.load to return the mocked state
        with patch('torch.load') as mock_torch_load:
            mock_torch_load.return_value = {
                'yolo_state_dict': {},
                'facenet_state_dict': {},
                'user_embeddings': {},
                'device': 'cpu',
                'yolo_model_path': 'yolov8n.pt'
            }

            # Load the combined model
            loaded_model = CombinedFacialRecognitionModel.load_model(combined_model_path)

        # Initialize FacialRecognition with the combined model
        with patch('myfacerec.facial_recognition.YOLO') as MockYOLO_inner:
            MockYOLO_inner.return_value = mock_yolo  # Reuse the same mock
            config = Config(user_data_path=str(tmp_path / "test_faces.json"))
            fr = FacialRecognition(config, combined_model_path=combined_model_path)

            # Mock the combined_model's forward method
            fr.combined_model.forward = MagicMock(return_value=[((10, 10, 50, 50), np.array([0.1] * 512, dtype=np.float32))])

    return fr

def test_register_with_face_separate_models(mock_facial_recognition):
    """
    Test registering a user using separate YOLO and Facenet models.
    """
    fr = mock_facial_recognition

    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("TestUser", [img])

    # Assertions
    assert "User 'TestUser' registered with 1 images." in msg
    assert fr.user_data.get("TestUser") is not None
    assert len(fr.user_data["TestUser"]) == 1
    # Ensure the mock was called correctly
    fr.detect_faces.assert_called_once_with(img)
    fr.embed_faces_batch.assert_called_once_with(img, [(10, 10, 50, 50)])

def test_register_with_face_combined_model(mock_combined_model):
    """
    Test registering a user using the combined YOLO and Facenet model.
    """
    fr = mock_combined_model

    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("TestUser", [img])

    # Assertions
    assert "User 'TestUser' registered with 1 images." in msg
    assert fr.combined_model.user_embeddings.get("TestUser") is not None
    assert len(fr.combined_model.user_embeddings["TestUser"]) == 1
    # Ensure the combined model's forward method was called
    fr.combined_model.forward.assert_called_once_with(img)

def test_identify_known_user_separate_models(mock_facial_recognition):
    """
    Test identifying a known user using separate YOLO and Facenet models.
    """
    fr = mock_facial_recognition

    # Pre-register a user
    fr.user_data = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    img = Image.new("RGB", (100, 100), color="white")
    results = fr.identify_user(img, threshold=0.5)

    # Assertions
    assert len(results) == 1
    assert results[0]['user_id'] == 'TestUser'
    assert results[0]['similarity'] == pytest.approx(1.0, abs=1e-6)
    # Ensure the mock was called correctly
    fr.detect_faces.assert_called_once_with(img)
    fr.embed_faces_batch.assert_called_once_with(img, [(10, 10, 50, 50)])

def test_identify_known_user_combined_model(mock_combined_model):
    """
    Test identifying a known user using the combined YOLO and Facenet model.
    """
    fr = mock_combined_model

    # Pre-register a user
    fr.combined_model.user_embeddings = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    img = Image.new("RGB", (100, 100), color="white")
    results = fr.identify_user(img, threshold=0.5)

    # Assertions
    assert len(results) == 1
    assert results[0]['user_id'] == 'TestUser'
    assert results[0]['similarity'] == pytest.approx(1.0, abs=1e-6)
    # Ensure the combined model's forward method was called
    fr.combined_model.forward.assert_called_once_with(img)

def test_export_import_model(tmp_path):
    """
    Test exporting and importing the combined model.
    """
    with patch('myfacerec.combined_model.YOLO') as MockYOLO:
        mock_yolo = MagicMock()
        mock_yolo.model.state_dict.return_value = {}
        MockYOLO.return_value = mock_yolo

        # Initialize combined model
        combined_model = CombinedFacialRecognitionModel(yolo_model_path="yolov8n.pt", device="cpu")
        combined_model.user_embeddings = {
            "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
        }

        # Save the combined model
        export_path = str(tmp_path / "exported_model.pt")
        combined_model.save_model(export_path)

        # Load the combined model
        loaded_model = CombinedFacialRecognitionModel.load_model(export_path)

        # Assertions
        assert loaded_model.user_embeddings == combined_model.user_embeddings
        # Ensure that the state dictionaries are equal
        for key in combined_model.yolo.model.state_dict():
            assert torch.equal(combined_model.yolo.model.state_dict()[key],
                               loaded_model.yolo.model.state_dict()[key])
        for key in combined_model.facenet.state_dict():
            assert torch.equal(combined_model.facenet.state_dict()[key],
                               loaded_model.facenet.state_dict()[key])

def test_export_model_separate_models(tmp_path, mock_facial_recognition):
    """
    Test exporting the model and user data when using separate YOLO and Facenet models.
    """
    fr = mock_facial_recognition

    # Pre-register a user
    fr.user_data = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    export_path = str(tmp_path / "exported_separate.pt")
    msg = fr.export_model(export_path)

    # Assertions
    assert "[Export] Model and user data saved to" in msg
    assert os.path.exists(export_path)

    # Load the exported state
    state = torch.load(export_path, map_location='cpu')
    assert 'yolo_state_dict' in state
    assert 'facenet_state_dict' in state
    assert 'user_embeddings' in state
    assert state['user_embeddings'] == fr.user_data

def test_export_model_combined_model(tmp_path, mock_combined_model):
    """
    Test exporting the combined model.
    """
    fr = mock_combined_model

    # Pre-register a user
    fr.combined_model.user_embeddings = {
        "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
    }

    with patch('myfacerec.combined_model.YOLO') as MockYOLO:
        mock_yolo = MagicMock()
        mock_yolo.model.state_dict.return_value = {}
        MockYOLO.return_value = mock_yolo

        export_path = str(tmp_path / "exported_combined.pt")
        msg = fr.export_model(export_path)

    # Assertions
    assert "[Export] Combined model saved to" in msg
    assert os.path.exists(export_path)

    # Load the exported combined model
    with patch('myfacerec.combined_model.YOLO') as MockYOLO_inner:
        mock_yolo_inner = MagicMock()
        mock_yolo_inner.model.state_dict.return_value = {}
        MockYOLO_inner.return_value = mock_yolo_inner

        loaded_model = CombinedFacialRecognitionModel.load_model(export_path)

    # Assertions
    assert loaded_model.user_embeddings == fr.combined_model.user_embeddings
    # Ensure that the state dictionaries are equal
    for key in fr.combined_model.yolo.model.state_dict():
        assert torch.equal(fr.combined_model.yolo.model.state_dict()[key],
                           loaded_model.yolo.model.state_dict()[key])
    for key in fr.combined_model.facenet.state_dict():
        assert torch.equal(fr.combined_model.facenet.state_dict()[key],
                           loaded_model.facenet.state_dict()[key])

def test_import_model(tmp_path):
    """
    Test importing a combined model.
    """
    with patch('myfacerec.combined_model.YOLO') as MockYOLO:
        mock_yolo = MagicMock()
        mock_yolo.model.state_dict.return_value = {}
        MockYOLO.return_value = mock_yolo

        # Initialize and save a combined model
        combined_model = CombinedFacialRecognitionModel(yolo_model_path="yolov8n.pt", device="cpu")
        combined_model.user_embeddings = {
            "TestUser": [np.array([0.1] * 512, dtype=np.float32)]
        }
        export_path = str(tmp_path / "import_model.pt")
        combined_model.save_model(export_path)

        # Patch torch.load to return the mocked state
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
            fr = FacialRecognition.import_model(export_path, config)

            # Assertions
            assert fr.combined_model.user_embeddings == combined_model.user_embeddings
            # Ensure that the state dictionaries are equal
            for key in combined_model.yolo.model.state_dict():
                assert torch.equal(combined_model.yolo.model.state_dict()[key],
                                   fr.combined_model.yolo.model.state_dict()[key])
            for key in combined_model.facenet.state_dict():
                assert torch.equal(combined_model.facenet.state_dict()[key],
                                   fr.combined_model.facenet.state_dict()[key])
