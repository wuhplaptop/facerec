Below is an **example** of a comprehensive **README.md** you can include with your project. It covers installation, usage instructions, an overview of each component, and credits. Feel free to edit sections (like your own name, repository links, or any additional instructions) as you see fit.

---

# Face Pipeline

A **single-file face recognition pipeline** built on YOLO, Mediapipe, DeepSORT, and FaceNet. It also integrates blink detection, anti-spoof, hand-tracking, eye-color detection, and a Gradio UI. This project allows you to detect faces, recognize known individuals, track them between frames, detect spoofing attempts, and more—all in a single minimal codebase.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Gradio Application Usage](#gradio-application-usage)
   - [Tabs Overview](#tabs-overview)
   - [Export / Import Config or Database](#export--import-config-or-database)
6. [API & Code Structure](#api--code-structure)
   - [1. `PipelineConfig`](#1-pipelineconfig)
   - [2. `FaceDatabase`](#2-facedatabase)
   - [3. `YOLOFaceDetector`](#3-yolofacedetector)
   - [4. `FaceTracker` (DeepSORT)](#4-facetracker-deepsort)
   - [5. `FaceNetEmbedder`](#5-facenetembedder)
   - [6. Blink Detection](#6-blink-detection)
   - [7. Face Mesh & Eye Color Detection](#7-face-mesh--eye-color-detection)
   - [8. `HandTracker`](#8-handtracker)
   - [9. `FacePipeline`](#9-facepipeline)
   - [10. Export/Import Helpers](#10-exportimport-helpers)
   - [11. Gradio `build_app` Function](#11-gradio-build_app-function)
7. [Testing](#testing)
8. [Credits](#credits)
9. [License](#license)

---

## Overview

This repository implements a **face recognition and tracking pipeline** in a single file (`face_pipeline.py`). It uses:

- **YOLO** (via Ultralytics) for face detection.
- **DeepSORT** for real-time multi-object tracking of faces.
- **FaceNet** (via `facenet-pytorch`) for face recognition embeddings.
- **Mediapipe** for:
  - **Blink detection** (Face Mesh)
  - **Eye color detection** (also uses Mediapipe Face Mesh)
  - **Hand tracking**
- **Anti-spoof** check based on image sharpness (Laplacian variance).
- **Gradio** for an easy-to-use **web interface**.

---

## Features

1. **Face Detection**  
   Uses a YOLO model (`face2.pt`) to detect faces in images or video frames.

2. **Face Tracking**  
   Uses **DeepSORT** to assign consistent IDs to faces across frames, so you can track them over time.

3. **Face Recognition**  
   Uses FaceNet embeddings to identify known individuals from a local “face database.”

4. **Anti-Spoof**  
   Checks if a detected face is “real” or a “spoof” using a blur/sharpness threshold (Laplacian variance).

5. **Blink Detection**  
   Uses Mediapipe’s Face Mesh landmarks to detect blinks based on Eye Aspect Ratio (EAR).

6. **Face Mesh (Contours / Irises / Tesselation)**  
   Uses Mediapipe Face Mesh to optionally draw face landmarks, iris landmarks, and more.

7. **Eye Color Detection**  
   Estimates the dominant color of the iris region using a basic KMeans color analysis, then maps it to a known color label.

8. **Hand Tracking**  
   Uses Mediapipe Hands to detect and annotate hands if enabled.

9. **Configurable**  
   All thresholds, toggles, bounding box colors, etc. are adjustable via the **Gradio** UI’s **Configuration** tab.

10. **Import / Export**  
    - Export/Import the entire face database as `.pkl` to share known faces with others.  
    - Export/Import the pipeline config (`.pkl`) to share the entire configuration.

---

## Installation

### 1. Clone or Download

```bash
git clone https://github.com/YourUsername/yourrepo.git
cd yourrepo
```

*(If you just have the files, place them all in one folder.)*

### 2. Install Package

Install in **editable mode** (local development):

```bash
pip install -e .
```

*(This looks for `setup.py` in the current directory, installs `face_pipeline`.)*

Or install from PyPI (assuming you’ve published it there):

```bash
pip install face-pipeline
```

### 3. Dependencies

All required dependencies (Torch, Mediapipe, ultralytics, etc.) are listed in `setup.py`. They will automatically install with `pip`. If you prefer, you can also install them from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Quick Start

1. **Launch the Gradio Interface**:

   ```bash
   face-pipeline
   ```

   or

   ```bash
   python -m face_pipeline
   ```

   - This starts a local web server on `http://0.0.0.0:7860`.

2. **Open your browser** at `http://localhost:7860`.
3. **Use the Tabs** to test images, configure thresholds, manage your database of known faces, etc.

---

## Gradio Application Usage

When you run `face-pipeline`, you’ll see a **Gradio UI** with several tabs:

### Tabs Overview

1. **Image Test**  
   Upload an image (single frame). The pipeline will detect faces, track them, recognize known identities, and highlight any spoofs or blinks. You’ll see an annotated output and JSON detection info.

2. **Configuration**  
   Turn features on/off (recognition, anti-spoof, blink, hand tracking, face mesh, etc.), adjust thresholds, and configure bounding-box colors. Then **Save** the config to persist changes to `~/.face_pipeline/config.pkl`.

3. **Database Management**  
   - **User Enrollment**: Provide a label and multiple face images to enroll a new user’s embeddings.  
   - **User Search**: Search by name or by uploading an image.  
   - **User Management Tools**: List all known user labels, remove specific users, refresh the list, etc.

4. **Export / Import**  
   - **Export** or **Import** the pipeline config (`.pkl`) for thresholds/colors.  
   - **Export** or **Import** the face embeddings database (`.pkl`) to share recognized faces.  
   - **Merge** or **overwrite** the local database on import.

### Export / Import Config or Database

- **Export Config**:  
  Specify a path (e.g., `my_config.pkl`), click **Export Config**. This dumps the pipeline thresholds and color settings.

- **Import Config**:  
  Upload or type the path to a `.pkl` config file, click **Import Config** to load it and re-initialize.

- **Export Database**:  
  Similar to config, but for the face embeddings. Creates, e.g., `my_database.pkl`.

- **Import Database**:  
  Select a `.pkl` file of face embeddings from someone else. Choose whether to **merge** with your existing DB or **overwrite** it entirely.

---

## API & Code Structure

Everything is in **`face_pipeline.py`**, which is a **single** large file containing:

### 1. `PipelineConfig`
A dataclass that holds all pipeline configuration:

- **`detector`**: model path, device (CPU vs GPU).  
- **`recognition`**: whether to enable face recognition, etc.  
- **`blink`**: blink detection thresholds.  
- **`anti_spoof`**: Laplacian variance threshold.  
- **Colors** for bounding boxes, face mesh lines, blink text, etc.

**Key Methods**:
- `save(path)` / `load(path)`: pickles/unpickles config to/from disk.  
- `export_config(...)` / `import_config(...)`: convenience for exporting/importing to `.pkl`.

### 2. `FaceDatabase`
Manages all known face embeddings:

- **`embeddings`**: a dict mapping `label -> List[np.ndarray of embeddings]`.
- **`add_embedding(label, embedding)`**: add an embedding for a user.
- **`remove_label(label)`**: delete a user entirely.
- **`list_labels()`**: list all known user labels.
- **`search_by_image(embedding, threshold)`**: returns matches with similarity >= threshold.

**Key Methods**:
- `save()`: pickles the entire embeddings dict to `~/.face_pipeline/known_faces.pkl`.
- `export_database(...)` / `import_database(...)`: for sharing or merging DB files.

### 3. `YOLOFaceDetector`
Wraps the **Ultralytics YOLO** model for face detection:

- Automatically downloads the custom `face2.pt` from GitHub if not present.
- **`detect(image, conf_thres)`**: returns bounding boxes, confidences, and classes.

### 4. `FaceTracker` (DeepSORT)
Implements multi-face tracking using **DeepSORT**:

- **`update(detections, frame)`**: returns a list of track objects with stable IDs, bounding boxes, etc.

### 5. `FaceNetEmbedder`
Uses **FaceNet** (via `facenet-pytorch`) to embed face crops into a 512-D vector:

- **`get_embedding(face_bgr)`**: returns a NumPy embedding or `None` on error.

### 6. Blink Detection
Via **Mediapipe Face Mesh** and Eye Aspect Ratio (EAR):

- **`detect_blink(face_roi, threshold=0.25)`**:
  - Returns a bool (blink or not), left & right EAR values, plus eye landmarks.

### 7. Face Mesh & Eye Color Detection
- **`process_face_mesh(face_roi)`** obtains Mediapipe landmarks for face mesh.
- **`draw_face_mesh(...)`** draws tesselation, contours, iris lines on the face image.
- **`detect_eye_color(face_roi, landmarks)`** does KMeans on the iris region to classify color as *blue*, *brown*, *green*, *hazel*, etc.

### 8. `HandTracker`
Uses **Mediapipe Hands** to detect, track, and draw landmarks for up to 2 hands:

- **`detect_hands(image)`** returns Mediapipe’s hand landmarks + handedness classification.
- **`draw_hands(image, landmarks, handedness, config)`** draws lines, circles, and text labels for each detected hand.

### 9. `FacePipeline`
The main pipeline class:

- **Holds** the config, the YOLO detector, the DeepSORT tracker, the FaceNet embedder, the FaceDatabase, and optionally the HandTracker.
- **`initialize()`** sets up everything, loading config from disk, models, etc.
- **`process_frame(frame)`** does the full detection + tracking + anti-spoof + recognition + blink + face mesh + hand detection in one pass. Returns an annotated frame and a list of detection results.

**Additional methods**:
- `is_real_face(face_roi)`: Laplacian-based anti-spoof check.  
- `recognize_face(embedding, threshold)`: returns best matching label or “Unknown.”

### 10. Export/Import Helpers
- **`export_config_file(export_path)`** / **`import_config_file(import_path)`**  
- **`export_db_file(export_path)`** / **`import_db_file(import_path, merge=True)`**  

Used by the Gradio UI to let users easily share config or known faces.

### 11. Gradio `build_app` Function
Defines the user interface:

- **Tabs**: “Image Test,” “Configuration,” “Database Management,” “Export / Import.”
- Each tab has UI elements (checkboxes, sliders, file upload, etc.) that call specific functions like `process_test_image` or `enroll_user`.

---

## Testing

A sample test file, `test_face_pipeline.py`, uses **pytest** to ensure:

1. You can import `face_pipeline` without errors.
2. `pipeline = face_pipeline.load_pipeline()` initializes successfully.

**Run tests**:

```bash
pytest
```

**Libraries & Frameworks**:  
1. [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – Face detection model backbone.  
2. [Deep SORT Realtime](https://github.com/GeekAlexander/deep_sort_realtime) – Realtime multi-object tracking.  
3. [facenet-pytorch](https://github.com/timesler/facenet-pytorch) – Pretrained FaceNet for embeddings.  
4. [Mediapipe](https://github.com/google/mediapipe) – Face Mesh, Hand Tracking, etc.  
5. [Gradio](https://github.com/gradio-app/gradio) – Web UI framework.  
6. [PyTorch](https://pytorch.org/) – Core deep-learning framework.  
7. [OpenCV](https://github.com/opencv/opencv) – Image processing.  
8. [Requests](https://docs.python-requests.org/) – For HTTP requests (downloading models).

**Other**:
- Face detection model `face2.pt` courtesy of [wuhplaptop/face-11-n](https://github.com/wuhplaptop/face-11-n).  
- This repository’s structure and CI/CD approach was inspired by standard Python packaging guidelines and GitHub Actions templates.

---
