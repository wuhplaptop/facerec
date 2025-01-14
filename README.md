# **rolo-rec**  
> *Future-proof Facial Recognition with YOLO + Facenet, modular detectors/embedders, hooks, CLI, etc.*

## **Table of Contents**

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [CLI Usage](#cli-usage)  
6. [Detailed Modules & Functions](#detailed-modules--functions)  
   1. [\_\_init\_\_.py](#1-__init__py)  
   2. [cli.py](#2-clipy)  
   3. [config.py](#3-configpy)  
   4. [facial_recognition.py](#4-facial_recognitionpy)  
   5. [detectors.py](#5-detectorspy)  
   6. [embedders.py](#6-embedderspy)  
   7. [data_store.py](#7-data_storepy)  
   8. [hooks.py](#8-hookspy)  
   9. [plugins/](#9-plugins)  
7. [Testing](#testing)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## **Overview**

**rolo-rec** is a modular facial recognition library built on **YOLO** (for face detection) and **Facenet** (for embedding and similarity checks). It provides:

- A **command-line interface** (`rolo-rec`) to register and identify faces.
- A **plug-and-play** architecture for detectors, embedders, data stores.
- **Hooks** that allow you to inject custom logic at various stages (detection, embedding, etc.).
- **Configurable** YOLO model paths and device selection (CPU/GPU).
- **JSON-based** data storage out of the box, with potential for plugin-based data store solutions.

---

## **Key Features**

1. **CLI for Easy Usage**  
   - **Register** faces with `rolo-rec register`.
   - **Identify** faces with `rolo-rec identify`.
   - **List, Import, Export** face embeddings data.

2. **Modular Architecture**  
   - **Detectors**: YOLO for face bounding boxes, or custom plugin detectors.
   - **Embedders**: Facenet for face embeddings, or custom plugin embedders.

3. **Hooks System**  
   - Register hooks (`before_detect`, `after_detect`, etc.) to apply transformations or logging at specific pipeline stages.

4. **Flexible Configuration**  
   - **`Config`** object sets YOLO model paths, confidence thresholds, alignment functions, plugin references, etc.

5. **Plugin Manager**  
   - Dynamically load custom detectors or embedders from your setup’s entry points.

---

## **Project Structure**

A typical `rolo-rec` project layout:

```
rolo-rec/
├── myfacerec/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data_store.py
│   ├── detectors.py
│   ├── embedders.py
│   ├── facial_recognition.py
│   ├── hooks.py
│   └── plugins/
│       ├── __init__.py
│       ├── base.py
│       └── sample_plugin.py
├── tests/
│   └── test_basic.py
├── setup.py
└── requirements.txt (optional)
```

---

## **Installation**

1. **Local Development**  
   ```bash
   git clone https://github.com/.../rolo-rec.git
   cd rolo-rec
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .  # Editable install
   ```

2. **Production Install from Source**  
   ```bash
   python setup.py sdist bdist_wheel
   pip install dist/rolo_rec-0.3.0-py3-none-any.whl
   ```

3. **PyPI (if published)**  
   ```bash
   pip install rolo-rec
   ```

---

## **CLI Usage**

Once installed, you can run:
```bash
rolo-rec --help
```
This provides help on subcommands. The main subcommands:

1. **register**  
   Register a user with one or more images:
   ```bash
   rolo-rec register --user Alice --images alice1.jpg alice2.jpg --conf 0.5
   ```
   - `--user`: Unique username.
   - `--images`: List of file paths to face images.
   - `--conf`: YOLO confidence threshold (default 0.5).

2. **identify**  
   Identify faces in an image:
   ```bash
   rolo-rec identify --image group_photo.jpg --threshold 0.6 --conf 0.5
   ```
   - `--image`: Path to the image containing faces.
   - `--threshold`: Similarity threshold for deciding if a face is recognized.

3. **export-data**  
   Export the JSON data (face embeddings) to a file:
   ```bash
   rolo-rec export-data --output shared_faces.json
   ```

4. **import-data**  
   Import face embeddings from a JSON file:
   ```bash
   rolo-rec import-data --file shared_faces.json
   ```

5. **list-users**  
   List all registered user IDs:
   ```bash
   rolo-rec list-users
   ```

6. **delete-user**  
   Delete a registered user from the data store:
   ```bash
   rolo-rec delete-user --user Alice
   ```

---

## **Detailed Modules & Functions**

Below is a **per-file** overview of each major class and function:

### **1. `__init__.py`**
Exports key classes so external code can do:
```python
from myfacerec import FacialRecognition, Config, YOLOFaceDetector, ...
```
- **`FacialRecognition`**: The main orchestrator class from `facial_recognition.py`.
- **`Config`**: The config class from `config.py`.
- **`YOLOFaceDetector`, `FacenetEmbedder`, `JSONUserDataStore`, `Hooks`**: Exposed for quick usage.

### **2. `cli.py`**
Implements the **rolo-rec** CLI:

- **`main()`**: Top-level entry point for subcommands.
- **Subparsers**:
  - **register**: Registers a user with images.
  - **identify**: Identifies faces in an image.
  - **export-data**/**import-data**: Manage JSON data export/import.
  - **list-users**/**delete-user**: Manage users.

**Key workflow**:
1. Parse arguments (`argparse`).
2. Create a `Config` object with the user’s specified thresholds, model paths, etc.
3. Construct a `FacialRecognition` object.
4. Call the relevant methods: `register_user(...)`, `identify_user(...)`, etc.

### **3. `config.py`**
Central config object, **`Config`**, that sets:
- **`yolo_model_path`**: Local or URL path to YOLO model.
- **`default_model_url`**: Backup YOLO model if none is specified.
- **`conf_threshold`**: YOLO confidence threshold.
- **`device`**: Defaults to “cuda” if GPU is available, else “cpu”.
- **`user_data_path`**: JSON file path for embeddings.
- **`alignment_fn`**: Optional function to align or transform images before embedding.
- **`before_detect`, `after_detect`, `before_embed`, `after_embed`**: Optional config-based hooks.
- **`detector_plugin`, `embedder_plugin`**: For plugin-based detectors/embedders.
- **`cache_dir`**: Base directory to cache downloaded YOLO models.

### **4. `facial_recognition.py`**
**Core** orchestrator class **`FacialRecognition`**. Key responsibilities:

1. **Constructor**:
   - Accepts `Config`, plus optional `detector`, `embedder`, `data_store`.
   - Loads YOLO if no custom detector is provided.
   - Loads FaceNet if no custom embedder is provided.
   - Loads a JSON user data store if none is provided.
   - Maintains a `self.user_data` dictionary from the data store.

2. **Face Detection** (`detect_faces`):
   - Calls the `FaceDetector` object’s `detect_faces(image)`.
   - Optionally calls hooks (`before_detect`, `after_detect`).

3. **Embedding** (`embed_faces_batch`):
   - Calls the `FaceEmbedder` object’s `embed_faces_batch(image, boxes)`.
   - Optionally calls hooks (`before_embed`, `after_embed`).

4. **Registering** (`register_user(user_id, images)`):
   - For each image, tries to detect exactly one face.
   - Embeds the face, stores the resulting vector(s) in `self.user_data[user_id]`.
   - Saves the updated user data to disk (JSON by default).

5. **Identifying** (`identify_user(image, threshold)`):
   - Detects faces, then for each face embedding, computes similarity vs. known embeddings.
   - If best similarity > threshold, recognized. Otherwise, “Unknown.”

6. **List / Delete / Update**:
   - `list_users()`, `delete_user(user_id)`, `update_user_embeddings(user_id, new_embeddings)`.

### **5. `detectors.py`**
- **`FaceDetector`** (abstract): Must implement `detect_faces(image)`.
- **`YOLOFaceDetector`**: Default YOLO-based face detection:
  1. Runs YOLO on the image.
  2. Filters out bounding boxes with `cls == 0` (person class) and `conf >= conf_threshold`.
  3. Returns a list of `(x1, y1, x2, y2)` bounding box tuples.

### **6. `embedders.py`**
- **`FaceEmbedder`** (abstract): Must implement `embed_faces_batch(image, boxes)`.
- **`FacenetEmbedder`**: 
  1. Crops and resizes each face to `(160,160)`.
  2. Optionally calls an `alignment_fn(face)` if provided.
  3. Normalizes the pixel data, converts to a PyTorch tensor.
  4. Passes the batch through FaceNet to get embeddings as a NumPy array.

### **7. `data_store.py`**
- **`UserDataStore`** (abstract).
- **`JSONUserDataStore`**:  
  - Reads/writes face embeddings in JSON (`user_faces.json` by default).
  - On load, reconstitutes embeddings as NumPy arrays.
  - On save, converts them to lists for JSON serialization.

### **8. `hooks.py`**
A **simple** hook management system:

- `Hooks.before_detect`, `Hooks.after_detect`, etc. store lists of callables.
- `Hooks.execute_before_detect(image)` calls each function in `before_detect`, passing the `image`.
- Useful for custom logic (e.g., image transformations, logging, metrics).

### **9. `plugins/`**
- **`base.py`**: Has a `PluginManager` class to dynamically load detectors, embedders, or data stores via `pkg_resources` entry points.
- **`sample_plugin.py`**: Example plugin showing a `SampleDetector` that does nothing (returns empty bounding boxes).  

---

## **Testing**

Your test suite in `tests/` typically includes:

- **`test_basic.py`**: Basic end-to-end tests for registration and identification:
  1. **`test_register_no_faces`**: Ensures the library handles images with no faces.
  2. **`test_register_with_face`**: Mocks face detection/embedding to confirm registration works.
  3. **`test_identify_unknown`**: Confirms that images with no recognized embeddings yield no results.
  4. **`test_identify_known_user`**: Mocks detection/embedding so a known user is recognized.
  5. **`test_plugin_loading`**: Ensures custom plugin detectors can be loaded.

**Running tests**:

```bash
pytest tests/
```

**Common Issues**:
- **AttributeError** (e.g., `'list' object has no attribute 'shape'`): Fix your mocks to return NumPy arrays.
- **Floating-Point** issues (e.g. `1.0000008344650269` vs. `1.0`): Use `pytest.approx(1.0, abs=1e-6)` to allow small numeric deviations.

---

## **Contributing**

1. **Fork & Clone**:  
   ```bash
   git clone https://github.com/YourUser/rolo-rec.git
   cd rolo-rec
   ```
2. **Create a Feature Branch**:  
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Install & Test**:  
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   pytest tests/
   ```
4. **Open a Pull Request**.

**Coding Guidelines**:
- Follow PEP8 for code style.
- Use docstrings for each function explaining arguments and return values.
- Write or update tests for new features or bugfixes.

---

## **License**

This project is distributed under the terms of the **MIT License** (or whichever license you choose). See [LICENSE](LICENSE) for details.

---

# **Summary**

- **rolo-rec** is a powerful, modular facial recognition library combining YOLO-based detection and FaceNet-based embeddings, offering a robust CLI and an extensible plugin system.
- The **`FacialRecognition`** class orchestrates the entire pipeline, from detection to embedding and user data management.
- Configuration is handled by **`Config`**, which includes crucial attributes like `alignment_fn`, device selection, YOLO paths, confidence thresholds, etc.
- The code is extensively **testable** via the `tests/` folder, using **pytest**.

We hope this README helps you get started quickly and effectively with **rolo-rec**! Feel free to open issues, contribute new plugins, or add enhancements like advanced data stores or alignment steps.
