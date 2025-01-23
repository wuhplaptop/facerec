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
# 2) Import code under test (AFTER the YOLO patch is in place)
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
    as Tensors. Enough for your code to do:
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

        # If you want to test pose, set this to True in a specific test
        self.enable_pose_estimation = False

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
    # Also confirm similarity is close to 1.0
    assert results[0]["similarity"] > 0.99


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
    # Check that we got some best_sim < 0.6
    assert results[0]["similarity"] < 0.6


def test_export_model(mock_facial_recognition, tmp_path):
    """
    Overriding yolo.model.state_dict() and facenet.model.state_dict() => real dict,
    plus ensuring yolo.model_path is a real str => no pickling error.
    We also wrap save_model in a patch to confirm it was called.
    """
    export_path = tmp_path / "exported_model.pt"
    mock_facial_recognition.user_data["user1"] = [np.array([0.1, 0.2, 0.3])]

    # Overwrite the real model state dict with a real dict
    mock_facial_recognition.model.yolo.model.state_dict = lambda: {}
    mock_facial_recognition.model.yolo.model_path = "fake_yolo_model.pt"

    # Now wrap the real 'save_model' so we can assert calls
    with patch.object(
        mock_facial_recognition.model,
        'save_model',
        wraps=mock_facial_recognition.model.save_model
    ) as mock_save:
        mock_facial_recognition.export_combined_model(str(export_path))
        mock_save.assert_called_once_with(str(export_path))

#
# ------------------------------------------------------------------
# OPTIONAL: Example test for pose estimation
# ------------------------------------------------------------------
#

def test_pose_estimation_enabled(mock_config, mock_data_store):
    """
    Demonstrates how you might test that 'pose' is computed
    (using the _dummy_landmark_detector) if enable_pose_estimation=True.
    """
    from myfacerec.facial_recognition import FacialRecognition
    from myfacerec.combined_model import CombinedFacialRecognitionModel

    # Enable pose
    mock_config.enable_pose_estimation = True

    # Build a fresh model/FR
    model = CombinedFacialRecognitionModel(
        yolo_model_path=mock_config.yolo_model_path,
        device=mock_config.device,
        conf_threshold=mock_config.conf_threshold,
        enable_pose_estimation=True
    )
    # Patch out the _dummy_landmark_detector or YOLO, etc.
    model.yolo.__call__.return_value = [FakeDetection([FakeBox()])]
    # Return some dummy embedding
    model.facenet.forward = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))

    # Create the FR object
    fr = FacialRecognition(config=mock_config, data_store=mock_data_store, model=model)

    # Now run identify or register
    result = fr.identify_user(MagicMock(spec=Image.Image))
    assert len(result) == 1
    # 'pose' should not be None if everything works
    assert result[0]['pose'] is not None, "Pose should be estimated when enable_pose_estimation=True."
