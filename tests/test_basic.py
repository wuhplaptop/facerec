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
    Globally patch `myfacerec.combined_model.YOLO` with a MagicMock that
    won't do actual file I/O. We'll inject fake detection objects ourselves.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_class:
        dummy_yolo = MagicMock()

        # If code does .to(...)
        dummy_yolo.to.return_value = dummy_yolo

        # If code does yolo.model.state_dict(), return a plain dict (picklable)
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}

        # If code calls yolo(...)
        dummy_yolo.__call__.return_value = []

        # If code tries to access yolo.model_path
        dummy_yolo.model_path = "fake_yolo_model.pt"

        yolo_class.return_value = dummy_yolo
        yield

# ------------------------------------------------------------------
# 2) Import your code after the YOLO patch
# ------------------------------------------------------------------
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# ------------------------------------------------------------------
# 3) Minimal "fake detection" classes to mimic ultralytics results
# ------------------------------------------------------------------
class FakeBox:
    """
    Minimal bounding box object that has .conf, .cls, .xyxy
    as Tensors. Enough for your real code to do:
        conf = box.conf.item()
        cls  = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0]
    """
    def __init__(self, conf=0.99, cls=0, xyxy=(10,10,100,100)):
        self.conf = torch.tensor([conf], dtype=torch.float)
        self.cls  = torch.tensor([float(cls)], dtype=torch.float)
        # shape [1,4]
        self.xyxy = torch.tensor([list(xyxy)], dtype=torch.float)

class FakeDetection:
    """
    Minimal detection object that has .boxes = [FakeBox, ...].
    """
    def __init__(self, boxes):
        # 'boxes' is a list of FakeBox
        self.boxes = boxes

# ------------------------------------------------------------------
# 4) Mock config
# ------------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"  # no real .pt load
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# ------------------------------------------------------------------
# 5) Construct the real CombinedFacialRecognitionModel
# ------------------------------------------------------------------
@pytest.fixture
def mock_combined_model(mock_config):
    """
    Returns a real CombinedFacialRecognitionModel, but YOLO constructor is patched.
    We'll also override facenet.model.state_dict so it's picklable.
    """
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold
    )

    # Make facenet.model.state_dict return a real dict
    model.facenet.model.state_dict = lambda: {}
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
    """
    Single face with conf=0.99, cls=0 => code should see it as a valid face,
    call facenet, and 'register' user with embedding [0.1,0.2,0.3].
    """
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # Build one FakeBox => conf=0.99, cls=0
    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])

    # Overwrite yolo(...) => returns [that detection]
    mock_facial_recognition.model.yolo.__call__.return_value = [detection]

    # Overwrite facenet => returns embedding [0.1,0.2,0.3]
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    """
    If embedding matches known_user => results[0]['user_id'] == known_user
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])
    mock_facial_recognition.model.yolo.__call__.return_value = [detection]

    # facenet => matching embedding
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition):
    """
    If embedding is dissimilar => 'Unknown'
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])
    mock_facial_recognition.model.yolo.__call__.return_value = [detection]

    # facenet => dissimilar
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.4, 0.5, 0.6]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, tmp_path):
    """
    Overriding yolo.model.state_dict() and facenet.model.state_dict() => real dict,
    plus ensuring yolo.model_path is a real str => no pickling error.
    """
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Overwrite real dict returning lambdas
    mock_facial_recognition.model.yolo.model.state_dict = lambda: {}
    mock_facial_recognition.model.yolo.model_path = "fake_yolo_model.pt"
    # Already done in fixture: model.facenet.model.state_dict = lambda: {}

    mock_facial_recognition.export_combined_model(str(export_path))
    # If we get here, no pickling error
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
