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
    Globally patch `myfacerec.combined_model.YOLO` so that calling YOLO(...) won't do
    real file I/O. We'll inject fake detection objects ourselves.
    """
    with patch("myfacerec.combined_model.YOLO") as yolo_class:
        # 1. Create a MagicMock instance to represent your YOLO object
        dummy_yolo = MagicMock()

        # 2. If code does something like yolo.to(...), have it return itself
        dummy_yolo.to.return_value = dummy_yolo

        # 3. If code does yolo.model.state_dict(), return a plain dict (picklable)
        dummy_yolo.model = MagicMock()
        dummy_yolo.model.state_dict.return_value = {}

        # 4. If code calls yolo(...) => we want it to return []
        #    Instead of dummy_yolo.__call__.return_value, do:
        dummy_yolo.return_value = []

        # 5. Whenever `YOLO()` is instantiated, return `dummy_yolo`
        yolo_class.return_value = dummy_yolo

        yield

# ------------------------------------------------------------------
# 2) Import your code under test
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
        # shape [1,4]
        self.xyxy = torch.tensor([list(xyxy)], dtype=torch.float)

class FakeDetection:
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
    user_id = "test_user"
    images = [MagicMock(spec=Image.Image)]

    # Build one FakeBox => conf=0.99, cls=0
    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])

    # Overwrite yolo(...) => returns [that detection]
    mock_facial_recognition.model.yolo.return_value = [detection]

    # Overwrite facenet => returns embedding [0.1,0.2,0.3]
    mock_facial_recognition.model.facenet.forward = MagicMock(
        return_value=torch.tensor([[0.1, 0.2, 0.3]])
    )

    msg = mock_facial_recognition.register_user(user_id, images)
    assert "registered" in msg.lower()
    assert user_id in mock_facial_recognition.user_data
    assert len(mock_facial_recognition.user_data[user_id]) == 1

def test_identify_user_known(mock_facial_recognition):
    user_id = "known_user"
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

    mock_facial_recognition.model.yolo.model.state_dict = lambda: {}
    mock_facial_recognition.model.yolo.model_path = "fake_yolo_model.pt"

    with patch.object(
        mock_facial_recognition.model,
        'save_model',
        wraps=mock_facial_recognition.model.save_model
    ) as mock_save:
        mock_facial_recognition.export_combined_model(str(export_path))
        mock_save.assert_called_once_with(str(export_path))

def test_pose_estimation_enabled(mock_config, mock_data_store):
    from myfacerec.facial_recognition import FacialRecognition
    from myfacerec.combined_model import CombinedFacialRecognitionModel

    # Enable pose
    mock_config.enable_pose_estimation = True

    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold,
        enable_pose_estimation=True
    )
    box = FakeBox(conf=0.99, cls=0, xyxy=(10,10,100,100))
    detection = FakeDetection([box])
    model.yolo.return_value = [detection]
    model.facenet.forward = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))

    fr = FacialRecognition(config=mock_config, data_store=mock_data_store, model=model)

    result = fr.identify_user(MagicMock(spec=Image.Image))
    assert len(result) == 1
    # 'pose' should be populated (or at least not crash)
    assert 'pose' in result[0], "Expected a 'pose' key in the detection result."
