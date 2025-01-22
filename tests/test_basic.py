import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# -----------------------------------------------------------------------------
# 1) Globally patch YOLO so it never attempts to load .pt files
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def patch_model_yolo():
    """
    Patch myfacerec.combined_model.YOLO with a MagicMock that won't do file I/O.
    Ensures 'fake_yolo_model.pt' isn't actually loaded.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_class:
        # Make yolo_class return a dummy YOLO instance
        dummy_yolo = MagicMock()
        # If code does dummy_yolo.to(...), just return dummy_yolo again
        dummy_yolo.to.return_value = dummy_yolo
        # A mock model so .model.state_dict() works
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}

        # The critical part: override __call__ with a MagicMock
        # so dummy_yolo(...) won't error about .return_value
        dummy_yolo.__call__ = MagicMock(return_value=[])

        # Whenever CombinedFacialRecognitionModel calls YOLO(...),
        # it gets dummy_yolo instead of the real constructor.
        yolo_class.return_value = dummy_yolo

        yield

# -----------------------------------------------------------------------------
# 2) Now import your classes after the patch fixture is set
# -----------------------------------------------------------------------------
from myfacerec.facial_recognition import FacialRecognition
from myfacerec.data_store import UserDataStore
from myfacerec.combined_model import CombinedFacialRecognitionModel

# -----------------------------------------------------------------------------
# 3) Config fixture
# -----------------------------------------------------------------------------
class MockConfig:
    def __init__(self):
        self.conf_threshold = 0.5
        self.similarity_threshold = 0.6
        self.user_data_path = "user_faces.json"
        self.yolo_model_path = "fake_yolo_model.pt"  # won't load .pt due to patch
        self.device = "cpu"
        self.cache_dir = "cache"
        self.alignment_fn = None

@pytest.fixture
def mock_config():
    return MockConfig()

# -----------------------------------------------------------------------------
# 4) Create a real CombinedFacialRecognitionModel, but patch means YOLO is faked
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_combined_model(mock_config):
    """
    Actually constructs CombinedFacialRecognitionModel, but the YOLO(...) call
    is already replaced by the patch above, so no real file loads.
    Then we override self.yolo, self.facenet as needed in each test.
    """
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold
    )
    # We'll do final overriding in each test
    model.yolo = MagicMock()
    model.facenet = MagicMock()
    model.user_embeddings = {}
    return model

# -----------------------------------------------------------------------------
# 5) Data store fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_data_store(mock_combined_model):
    data_store = MagicMock(spec=UserDataStore)
    # tie data store to model.user_embeddings
    data_store.load_user_data.return_value = mock_combined_model.user_embeddings
    data_store.save_user_data.return_value = None
    return data_store

# -----------------------------------------------------------------------------
# 6) FacialRecognition fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_facial_recognition(mock_config, mock_combined_model, mock_data_store):
    return FacialRecognition(
        config=mock_config,
        data_store=mock_data_store,
        model=mock_combined_model
    )

# -----------------------------------------------------------------------------
# 7) Tests
# -----------------------------------------------------------------------------

def test_register_user(mock_facial_recognition):
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # YOLO => one bounding box
    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0  # class=0 => face
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]

    # So self.model.yolo(...) returns that single detection
    mock_facial_recognition.model.yolo.return_value = [detection_mock]

    # Facenet => single embedding
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower(), f"Expected registration success, got: {msg}"
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    user_id = "known_user"
    # Known user has embedding [0.1,0.2,0.3]
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]

    mock_facial_recognition.model.yolo.return_value = [detection_mock]
    # Return the same embedding => recognized
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == user_id

def test_identify_user_unknown(mock_facial_recognition):
    user_id = "known_user"
    mock_facial_recognition.user_data[user_id] = [np.array([0.1, 0.2, 0.3])]

    detection_mock = MagicMock()
    box_mock = MagicMock()
    box_mock.conf.item.return_value = 0.99
    box_mock.cls.item.return_value = 0
    box_mock.xyxy = [torch.tensor([10, 10, 100, 100], dtype=torch.float)]
    detection_mock.boxes = [box_mock]

    mock_facial_recognition.model.yolo.return_value = [detection_mock]
    # Dissimilar embedding => "Unknown"
    mock_facial_recognition.model.facenet.return_value = torch.tensor([[0.4, 0.5, 0.6]])

    results = mock_facial_recognition.identify_user(MagicMock(spec=Image.Image))
    assert len(results) == 1
    assert results[0]["user_id"] == "Unknown"

def test_export_model(mock_facial_recognition, tmp_path):
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Just ensure no exception & confirm we call save_model
    mock_facial_recognition.export_combined_model(str(export_path))
    mock_facial_recognition.model.save_model.assert_called_once_with(str(export_path))
