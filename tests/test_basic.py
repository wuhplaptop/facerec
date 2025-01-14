# tests/test_basic.py

import os
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.combined_model import CombinedFacialRecognitionModel
import torch

@pytest.fixture
def mock_facial_recognition(tmp_path):
    """
    Fixture to create a FacialRecognition instance using a CombinedFacialRecognitionModel with mocked embeddings.
    """
    with patch('myfacerec.combined_model.YOLO') as MockYOLO, \
         patch('myfacerec.combined_model.InceptionResnetV1') as MockResnet:

        # Mock YOLO
        mock_yolo = MagicMock()
        mock_yolo.model.state_dict.return_value = {}  # Ensure it returns a dict
        MockYOLO.return_value = mock_yolo

        # Mock InceptionResnetV1
        mock_facenet = MagicMock()
        MockResnet.return_value = mock_facenet
        mock_facenet.state_dict.return_value = {}  # Ensure it returns a dict
        mock_facenet.load_state_dict.return_value = None  # Mock load_state_dict to do nothing

        # Initialize the combined model
        combined_model = CombinedFacialRecognitionModel(yolo_model_path="yolov8n.pt", device="cpu")
        combined_model.user_embeddings = {}  # Start with no users

        # Mock detect_faces and embed_faces_batch methods
        combined_model.yolo.detect_faces = MagicMock(return_value=[(10, 10, 50, 50)])
        combined_model.facenet.embed_faces_batch = MagicMock(return_value=np.array([0.1] * 512, dtype=np.float32))

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
        config = Config(user_data_path=str(tmp_path / "test_faces.json"))
        fr = FacialRecognition(config, combined_model_path=combined_model_path)

        # Mock detect_faces and embed_faces_batch methods in FacialRecognition
        fr.combined_model.yolo.detect_faces = MagicMock(return_value=[(10, 10, 50, 50)])
        fr.combined_model.facenet.embed_faces_batch = MagicMock(return_value=np.array([0.1] * 512, dtype=np.float32))

    return fr
