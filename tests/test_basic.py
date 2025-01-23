import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# ------------------------------------------------------------------
# 1) Patch YOLO so it never tries to load a real .pt file
# ------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def patch_model_yolo():
    """
    Globally patch `myfacerec.combined_model.YOLO` so that calling YOLO(...)
    won't do real file I/O. We'll inject fake detection objects ourselves.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_class:
        dummy_yolo = MagicMock()

        # If code does yolo.to(...)
        dummy_yolo.to.return_value = dummy_yolo

        # If code does yolo.model.state_dict(), return a plain dict
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}

        # If code calls yolo(...)
        dummy_yolo.return_value = []

        # YOLO() constructor returns this mock
        yolo_class.return_value = dummy_yolo
        yield

# ------------------------------------------------------------------
# 2) Import code under test
# ------------------------------------------------------------------
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# ------------------------------------------------------------------
# 3) Minimal "fake detection" classes to mimic ultralytics results
# ------------------------------------------------------------------
class FakeBox:
    def __init__(self, conf=0.99, cls=0, xyxy=(10,10,100,100)):
        self.conf = torch.tensor([conf], dtype=torch.float)
        self.cls  = torch.tensor([float(cls)], dtype=torch.float)
        self.xyxy = torch.tensor([list(xyxy)], dtype=torch.float)

class FakeDetection:
    def __init__(self, boxes):
        self.boxes = boxes  # list of FakeBox

# ------------------------------------------------------------------
# 4) Mock config
# ------------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None
        self.enable_pose_estimation = False

@pytest.fixture
def mock_config():
    return MockConfig()

# ------------------------------------------------------------------
# 5) Construct the real CombinedFacialRecognitionModel
# ------------------------------------------------------------------
@pytest.fixture
def mock_combined_model(mock_config):
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold,
        enable_pose_estimation=mock_config.enable_pose_estimation
    )
    # Instead of model.facenet.model, we do:
    model.facenet.state_dict = lambda: {}
    return model

# ------------------------------------------------------------------
# 6) Data store fixture
# ------------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    ds = MagicMock(spec=UserDataStore)
    ds.load_user_data.return_value = mock_combined_model.user_embeddings
    ds.save_user_data.return_value = None
    return ds

# ------------------------------------------------------------------
# 7) FacialRecognition fixture
# ------------------------------------------------------------------
@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    return FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )

# ------------------------------------------------------------------
# 8) Tests
# ------------------------------------------------------------------

def test_register_user(mock_facial_recognition):
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # Create a single fake box
    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])

    # Mock yolo(...) => returns [that detection]
    mock_facial_recognition.model.yolo.return_value = [detection]

    # Mock FaceNet => returns a known embedding
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower()
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    user_id = "known_user"
    # Already in data store
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])
    mock_facial_recognition.model.yolo.return_value = [detection]

    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == user_id
    assert results[0]["similarity"] > 0.99

def test_identify_user_unknown(mock_facial_recognition):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])
    mock_facial_recognition.model.yolo.return_value = [detection]

    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.4, 0.5, 0.6]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"
    assert results[0]["similarity"] < 0.6

def test_export_model(mock_facial_recognition, tmp_path):
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Overwrite YOLO's state_dict
    mock_facial_recognition.model.yolo.model.state_dict = lambda: {}
    mock_facial_recognition.model.yolo.model_path = "fake_yolo_model.pt"

    # We'll patch 'save_model' to ensure it was called
    with patch.object(
        mock_facial_recognition.model,
        'save_model',
        wraps=mock_facial_recognition.model.save_model
    ) as mock_save:
        mock_facial_recognition.export_combined_model(str(export_path))
        mock_save.assert_called_once_with(str(export_path))

def test_pose_estimation_enabled(mock_config, mock_data_store):
    # Enable pose
    mock_config.enable_pose_estimation = True

    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold,
        enable_pose_estimation=True
    )
    # Patch YOLO & FaceNet
    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])
    model.yolo.return_value = [detection]
    model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    from myfacerec.facial_recognition import FacialRecognition
    fr = FacialRecognition(config=mock_config, data_store=mock_data_store, model=model)

    results = fr.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert 'pose' in results[0], "Expected a 'pose' key in the result when pose is enabled."
