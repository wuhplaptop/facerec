# tests/test_basic.py
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# 1) We import the real YOLO 'Results' and 'Boxes' classes
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.engine.results import Boxes

# -----------------------------------------------------------------
# 1) Patch YOLO in myfacerec.combined_model so it won't do file IO
# -----------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def patch_model_yolo():
    """
    Patch myfacerec.combined_model.YOLO with a MagicMock so it never actually
    loads a .pt file. We'll still replace the detection results with real
    'Results' objects in each test.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_class:
        dummy_yolo = MagicMock()
        # If code does .to(...), just return dummy_yolo
        dummy_yolo.to.return_value = dummy_yolo
        # If code does .model.state_dict(), default return is empty dict
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}

        # If code calls dummy_yolo(image), by default return []
        dummy_yolo.__call__ = MagicMock(return_value=[])

        # Make sure model_path is a real string (not MagicMock)
        dummy_yolo.model_path = "fake_yolo_model.pt"

        yolo_class.return_value = dummy_yolo
        yield

# Now import the code after the patch is in place
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# -----------------------------------------------------------------
# 2) Mock Config
# -----------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# -----------------------------------------------------------------
# 3) Real CombinedFacialRecognitionModel
# -----------------------------------------------------------------
@pytest.fixture
def mock_combined_model(mock_config):
    """
    Returns the real CombinedFacialRecognitionModel, but YOLO is patched above.
    We'll produce real 'Results' objects in each test so forward(...) sees them.
    """
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold
    )

    # Also set model.facenet.model.state_dict => empty dict (prevents pickling mocks)
    model.facenet.model.state_dict = lambda: {}
    return model

# -----------------------------------------------------------------
# 4) Data store fixture
# -----------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    ds = MagicMock(spec=UserDataStore)
    ds.load_user_data.return_value = mock_combined_model.user_embeddings
    ds.save_user_data.return_value = None
    return ds

# -----------------------------------------------------------------
# 5) FacialRecognition fixture
# -----------------------------------------------------------------
@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    return FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )

# -----------------------------------------------------------------
# Utility: build a real YOLO 'Results' object with conf, cls, box
# -----------------------------------------------------------------
def make_detection(conf=0.99, cls=0, xyxy=(10,10,100,100)):
    """
    Creates a real ultralytics 'Results' object with .boxes = real 'Boxes'
    containing shape Nx6 => [x1, y1, x2, y2, conf, cls].
    """
    x1,y1,x2,y2 = xyxy
    # shape [1,6]
    data = torch.tensor([[x1, y1, x2, y2, conf, float(cls)]], dtype=torch.float)

    # Create a real 'Boxes' object
    b = Boxes(data)

    # Put that inside a real 'Results' object
    r = Results()
    r.boxes = b
    return r

# -----------------------------------------------------------------
# 6) Tests
# -----------------------------------------------------------------

def test_register_user(mock_facial_recognition):
    """
    One real detection => should yield "registered".
    """
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # Build a real detection with conf=0.99, cls=0 => face
    detection = make_detection(conf=0.99, cls=0, xyxy=(10,10,100,100))
    # Override YOLO => returns [detection]
    mock_facial_recognition.model.yolo.__call__.return_value = [detection]

    # Facenet => real [0.1,0.2,0.3] embedding
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    """
    If embedding is close => recognized as 'known_user'.
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection = make_detection(conf=0.99, cls=0, xyxy=(10,10,100,100))
    mock_facial_recognition.model.yolo.__call__.return_value = [detection]

    # Facenet => matching
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition):
    """
    If embedding is dissimilar => 'Unknown'.
    """
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection = make_detection(conf=0.99, cls=0, xyxy=(10,10,100,100))
    mock_facial_recognition.model.yolo.__call__.return_value = [detection]

    # Facenet => dissimilar
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.4, 0.5, 0.6]])
    )

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1, f"Expected 1 face, got {len(results)}"
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, tmp_path):
    """
    Overriding yolo.model.state_dict() and facenet.state_dict() => real dict
    ensures no pickling errors. Also set yolo.model_path to a real string.
    """
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Ensure it won't try to pickle a MagicMock
    mock_facial_recognition.model.yolo.model.state_dict = lambda: {}
    mock_facial_recognition.model.yolo.model_path = "fake_yolo_model.pt"
    mock_facial_recognition.model.facenet.state_dict = lambda: {}

    mock_facial_recognition.export_combined_model(str(export_path))
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
