# rolo-rec

A modular **face recognition** library that uses **YOLO** (via [ultralytics](https://pypi.org/project/ultralytics/)) for face detection and **Facenet** ([facenet-pytorch](https://pypi.org/project/facenet-pytorch/)) for facial embeddings.  
Despite being published under the **distribution name** `rolo-rec`, the **import path** in Python code is `myfacerec.*`.  

## 1. Installation

```bash
pip install rolo-rec
```

- Requires **Python 3.7+**.
- Installs `myfacerec` as the internal package namespace, but the library is published under `rolo-rec`.

## 2. Usage

### 2.1 Command-Line Interface

When you install **rolo-rec**, it provides a CLI command called **`rolo-rec`** (if configured in your `setup.py` `entry_points`). You can:

1. **Register** a user:
   ```bash
   rolo-rec register --user Alice --images alice1.jpg alice2.jpg --conf 0.75
   ```
   - **`--user Alice`**: The unique user identifier.
   - **`--images alice1.jpg alice2.jpg`**: One or more paths to images containing that user’s face.
   - **`--conf 0.75`**: Optional YOLO detection confidence threshold (default 0.5).

2. **Identify** faces:
   ```bash
   rolo-rec identify --image group_photo.jpg --threshold 0.65 --conf 0.6
   ```
   - **`--image group_photo.jpg`**: The image in which to detect faces.
   - **`--threshold 0.65`**: Cosine similarity threshold for deciding if a face is recognized.
   - **`--conf 0.6`**: YOLO detection confidence threshold.

Sample output:
```
Face 1: box=(50, 40, 120, 120), user_id='Alice', similarity=0.72
Face 2: box=(130, 50, 180, 120), user_id='Unknown', similarity=0.55
```

### 2.2 Python API

You can also import the library in your own Python scripts. Even though you installed `rolo-rec`, **the import path** is `myfacerec`:

```python
from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition
from PIL import Image

# 1. Create a Config object, customizing thresholds or data paths
config = Config(
    user_data_path="user_faces.json",  # default storage for embeddings
    conf_threshold=0.75               # YOLO detection confidence threshold
)

# 2. Initialize the main class
fr = FacialRecognition(config)

# 3. Register a user with multiple images
img1 = Image.open("alice1.jpg").convert("RGB")
img2 = Image.open("alice2.jpg").convert("RGB")
message = fr.register_user("Alice", [img1, img2])
print(message)  # e.g. "[Registration] User 'Alice' registered with 2 images."

# 4. Identify faces in another image
test_img = Image.open("group_photo.jpg").convert("RGB")
results = fr.identify_user(test_img, threshold=0.65)

for i, r in enumerate(results):
    print(f"Face {i+1}: box={r['box']}, user_id={r['user_id']}, similarity={r['similarity']:.2f}")
```

**Notes**:
- **`conf_threshold`** (YOLO detection) and `threshold` (for face recognition similarity) are separate. 
- By default, embeddings are stored in a JSON file (`user_faces.json`). Each user can have multiple stored embeddings.

## 3. Features and Architecture

- **YOLO-based Face Detection**  
  Uses a `.pt` model for face detection via [ultralytics](https://pypi.org/project/ultralytics/).  
- **Facenet Embedding**  
  [facenet-pytorch](https://pypi.org/project/facenet-pytorch/) computes a 512-d vector for each face.  
- **Pluggable Design**  
  The library has separate modules for detection, embedding, and data storage. You can swap them for your own custom backends by creating new classes.  
- **JSON Data Store**  
  By default, user embeddings are stored in `user_faces.json`.  
- **Multiple Embeddings per User**  
  In `register_user()`, each image with exactly one face is converted to an embedding. They accumulate for robust recognition.  

## 4. Testing

If you cloned the source repository, you’ll find a `tests/` folder. For example:

```python
# tests/test_basic.py

import os
import pytest
from PIL import Image
from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition

def test_register_no_faces(tmp_path):
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path, conf_threshold=0.99)  # Very high => no detection
    fr = FacialRecognition(config)

    # Create a blank image
    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("NoFaceUser", [img])

    assert "No valid single-face images" in msg
    assert not fr.user_data.get("NoFaceUser")
```

You can run the tests using:

```bash
pytest tests/
```

## 5. Summary

**rolo-rec** is published under that PyPI package name, but when you import in code, it appears as **`myfacerec.*`**. The CLI command is also `rolo-rec` if you configured it in your `setup.py` entry points. 

- Install: `pip install rolo-rec`
- CLI usage: `rolo-rec register ...`, `rolo-rec identify ...`
- Python usage: `from myfacerec.facial_recognition import FacialRecognition`  

This flexible approach ensures you can easily register new users, detect faces in images, and integrate advanced face recognition capabilities into your projects.
