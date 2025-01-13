Below is an example **README** for a package named **`rolo-rec`**. You can copy-paste this into a file named `README.md` in the root of your repository. It covers:

1. **Introduction**  
2. **Installation**  
3. **Command-Line Usage**  
4. **Python API Usage**  
5. **Features & Architecture**  
6. **Testing**

Feel free to customize wording or add/remove sections as you see fit.

---

# rolo-rec

A modular **face recognition** library that uses **YOLO** for face detection and **Facenet** for embeddings. This library supports multiple face embeddings per user, a configurable detection threshold, and a JSON-based data store by default. It’s designed for easy customization, allowing you to swap detection or embedding backends if needed.

---

## 1. Installation

You can install this package from PyPI (once it’s published) using:

```bash
pip install rolo-rec
```

Or, if you have a local copy of the source code:

```bash
pip install .
```

> **Note**:  
> - Requires **Python 3.7+**  
> - If you intend to run YOLO or Facenet on a GPU, ensure you have a CUDA-capable PyTorch installed.

---

## 2. Command-Line Usage

**`rolo-rec`** provides a convenient CLI for registering and identifying faces. 

### 2.1 Register a User

```bash
rolo-rec register --user Alice --images alice1.jpg alice2.jpg --conf 0.75
```
- **`--user Alice`**: Unique identifier for the user.  
- **`--images alice1.jpg alice2.jpg`**: One or more image paths containing that user’s face.  
- **`--conf 0.75`**: (Optional) Confidence threshold for YOLO-based face detection (default is 0.5).  

Images with exactly one detected face will be embedded and stored in a JSON file (`user_faces.json` by default).

### 2.2 Identify Faces

```bash
rolo-rec identify --image group_photo.jpg --threshold 0.65 --conf 0.6
```
- **`--image group_photo.jpg`**: Path to the image to search for faces.  
- **`--threshold 0.65`**: Cosine similarity threshold for matching.  
- **`--conf 0.6`**: YOLO detection confidence threshold.

Output might look like:

```
Face 1: box=(50, 40, 120, 120), user_id='Alice', similarity=0.72
Face 2: box=(130, 50, 180, 120), user_id='Unknown', similarity=0.55
```

---

## 3. Python API Usage

If you prefer to integrate **`rolo-rec`** directly into your Python code:

```python
from rolo_rec.config import Config
from rolo_rec.facial_recognition import FacialRecognition
from PIL import Image

# 1. Create a config (customizing YOLO conf threshold, etc.)
config = Config(
    conf_threshold=0.75,
    user_data_path="user_faces.json"
)

# 2. Initialize the orchestrator
fr = FacialRecognition(config)

# 3. Register a user
img1 = Image.open("alice1.jpg").convert("RGB")
img2 = Image.open("alice2.jpg").convert("RGB")
msg = fr.register_user("Alice", [img1, img2])
print(msg)  # e.g. "[Registration] User 'Alice' registered with 2 images."

# 4. Identify in a new image
test_img = Image.open("group_photo.jpg").convert("RGB")
results = fr.identify_user(test_img, threshold=0.65)

for r in results:
    print(r)
    # {'box': (x1, y1, x2, y2), 'user_id': 'Alice', 'similarity': 0.72} etc.
```

---

## 4. Features & Architecture

1. **YOLO Face Detector**  
   - Uses the [ultralytics](https://pypi.org/project/ultralytics/) library to load a YOLO `.pt` model for face detection.  
   - Confidence threshold is configurable (`conf_threshold`).  

2. **Facenet Embedder**  
   - [facenet-pytorch](https://pypi.org/project/facenet-pytorch/) for generating 512-dimensional embeddings.  
   - Optionally supports face alignment or custom preprocessing hooks.  

3. **JSON Data Storage**  
   - Embeddings are stored by user ID in a JSON file.  
   - Easily swap for another storage solution by implementing a new class (e.g., a database).  

4. **Config & Hooks**  
   - `Config` includes fields like `yolo_model_path`, `device`, and optional callback hooks (`before_detect`, `after_detect`, etc.).  
   - Hooks allow you to customize or log each detection/embedding step.  

5. **Multiple Embeddings per User**  
   - Each user can register with many images, capturing different angles/lighting for more robust recognition.

---

## 5. Testing

If you’ve cloned the source repository, you can run the tests with:

```bash
pytest tests/
```

**Example Test** (`test_basic.py`):
```python
def test_register_no_faces(tmp_path):
    user_data_path = str(tmp_path / "test_faces.json")
    config = Config(user_data_path=user_data_path, conf_threshold=0.99)  # high => probably no detect
    fr = FacialRecognition(config)

    # blank 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    msg = fr.register_user("NoFaceUser", [img])

    assert "No valid single-face images" in msg
```

---

## 6. Summary

- **`rolo-rec`** provides a flexible, pluggable approach to face recognition.  
- For **quick usage**, rely on the CLI: `rolo-rec register` and `rolo-rec identify`.  
- For **advanced integration**, use the Python API and pass custom detectors, embedders, or data stores.  
- Set **thresholds** and **hooks** in `Config` to adapt to various conditions or user flows.  

---

**Enjoy building face recognition solutions with `rolo-rec`!**
