# FaceRec: Comprehensive Face Recognition Pipeline

![Face Recognition](https://github.com/wuhplaptop/facerec/blob/main/images/face_recognition_banner.png?raw=true)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Launching the Gradio App](#launching-the-gradio-app)
  - [Image Test](#image-test)
  - [Configuration](#configuration)
  - [Database Management](#database-management)
  - [Export / Import](#export--import)
- [How It Works](#how-it-works)
  - [Detection](#detection)
  - [Tracking](#tracking)
  - [Recognition](#recognition)
  - [Anti-Spoofing](#anti-spoofing)
  - [Blink Detection](#blink-detection)
  - [Eye Color Detection](#eye-color-detection)
  - [Face Mesh](#face-mesh)
  - [Hand Tracking](#hand-tracking)
  - [Export/Import Mechanism](#exportimport-mechanism)
- [Packages and Credits](#packages-and-credits)
- [License](#license)

## Overview

Welcome to **FaceRec**, a comprehensive face recognition pipeline that integrates state-of-the-art technologies for face detection, tracking, recognition, and analysis. Leveraging powerful libraries like **YOLO**, **FaceNet**, **Deep SORT**, and **Mediapipe**, FaceRec provides a robust and flexible solution for real-time face analysis. The user-friendly **Gradio** interface ensures ease of use, allowing users to interact with the system seamlessly.

## Features

- **Face Detection**: Utilizes YOLO for precise and efficient face detection.
- **Face Tracking**: Implements Deep SORT for reliable tracking of detected faces across frames.
- **Face Recognition**: Employs FaceNet for generating and comparing face embeddings to recognize individuals.
- **Anti-Spoofing**: Detects potential spoofed faces to ensure authenticity.
- **Blink Detection**: Monitors eye aspect ratios to identify blinks.
- **Eye Color Detection**: Analyzes iris regions to determine eye color.
- **Face Mesh**: Integrates Mediapipe's face mesh for detailed facial landmark analysis.
- **Hand Tracking**: Uses Mediapipe Hands for real-time hand detection and tracking.
- **Gradio Interface**: Provides an intuitive web-based UI for testing, configuration, database management, and export/import operations.
- **Combined Export/Import**: Allows users to export and import both configuration settings and the face database in a single file.

## Installation

FaceRec can be easily installed via `pip`. Ensure you have **Python 3.7** or higher installed on your system.

### Prerequisites

- **Python 3.7 or higher**
- **pip** (Python package installer)
- **Git** (optional, for cloning the repository)

### Steps

1. **Clone the Repository (Optional)**
   
   If you prefer to clone the repository, use the following commands:
   
   ```bash
   git clone https://github.com/wuhplaptop/facerec.git
   cd facerec
   ```

2. **Install via pip**
   
   Install FaceRec directly using `pip`:
   
   ```bash
   pip install face-pipeline
   ```

3. **Download YOLO Model (Automatic)**
   
   The system automatically downloads the YOLO face detection model if it's not present in the specified directory during the first run.

## Usage

FaceRec provides a powerful Gradio-based web interface for interacting with the system. Below are detailed instructions on how to use each feature.

### Launching the Gradio App

After installation, launch the Gradio app using the following command:

```bash
python -m face_pipeline
```

*Replace `face_pipeline` with the actual module name if different.*

Once launched, open your browser and navigate to `http://0.0.0.0:7860` to access the application.

### Image Test

**Purpose**: Upload a single image to perform face detection, recognition, blink detection, face mesh analysis, and hand tracking.

**Steps**:
1. Navigate to the **Image Test** tab.
2. Click on **Upload Image** to select an image from your device.
3. Click **Process Image** to run the analysis.
4. View the annotated image and detection results.

![Image Test](https://github.com/wuhplaptop/facerec/blob/main/images/image_test.png?raw=true)

### Configuration

**Purpose**: Customize detection parameters, toggle features, and adjust color settings to tailor the pipeline to your needs.

**Features**:
- **Toggle Components**: Enable or disable recognition, anti-spoofing, blink detection, hand tracking, eye color detection, and face mesh.
- **Face Mesh Options**: Toggle tesselation, contours, and irises display.
- **Threshold Settings**: Adjust detection confidence, recognition threshold, anti-spoof threshold, blink EAR threshold, hand detection confidence, and hand tracking confidence.
- **Color Customization**: Set colors for bounding boxes, eye outlines, blink text, hand landmarks, connections, mesh, contours, irises, and eye color text.

**Steps**:
1. Navigate to the **Configuration** tab.
2. Adjust the desired settings using checkboxes, sliders, and text inputs for color hex codes.
3. Click **Save Configuration** to apply and persist changes.

![Configuration](https://github.com/wuhplaptop/facerec/blob/main/images/configuration.png?raw=true)

### Database Management

**Purpose**: Manage the face database by enrolling new users, searching, listing, and removing existing entries.

#### User Enrollment

- **Enroll a New User**: Provide a user name and upload multiple images to add to the database.

#### User Search

- **Search By Name**: Select a user from the dropdown to retrieve their embeddings.
- **Search By Image**: Upload an image to find matching faces in the database.

#### User Management Tools

- **List Enrolled Users**: View all users currently enrolled in the database.
- **Remove User**: Select a user from the dropdown and remove them from the database.

**Steps**:
1. Navigate to the **Database Management** tab.
2. Expand the desired accordion section (**User Enrollment**, **User Search**, or **User Management Tools**) to perform actions.
3. Follow on-screen instructions to enroll, search, list, or remove users.

![Database Management](https://github.com/wuhplaptop/facerec/blob/main/images/database_management.png?raw=true)

### Export / Import

**Purpose**: Export and import both the configuration settings and face database either individually or combined into a single file.

#### Export Individually (Server Paths)

- **Export Config**: Specify a server path to save the current configuration.
- **Export Database**: Specify a server path to save the face database.

#### Import Individually (Server Paths)

- **Import Config**: Upload a configuration file from your device to apply settings.
- **Import Database**: Upload a database file from your device to merge or overwrite existing data.

#### Export & Import All Together

- **Export All**: Download a combined file containing both configuration and database.
- **Import All**: Upload a combined file to apply both configuration and database settings simultaneously.

**Steps**:
1. Navigate to the **Export / Import** tab.
2. Choose between exporting/importing individually or using the combined option.
3. Follow the on-screen prompts to perform the desired action.
4. For downloading exported files, a download link will be provided directly in the browser.

![Export Import](https://github.com/wuhplaptop/facerec/blob/main/images/export_import.png?raw=true)

## How It Works

The FaceRec system integrates multiple components to deliver a seamless and robust face analysis experience. Below is an in-depth explanation of each component and its role in the system.

### Detection

**YOLO (You Only Look Once)** is employed for real-time face detection. It processes input images to identify bounding boxes around faces with associated confidence scores.

- **Model Loading**: Automatically downloads the YOLO model if not present.
- **Detection Process**: Scans the image and returns coordinates for detected faces.

### Tracking

**Deep SORT (Simple Online and Realtime Tracking)** is utilized to maintain consistent tracking of detected faces across multiple frames or images.

- **Initialization**: Configured with parameters like `max_age` to determine how long to keep tracking a face.
- **Tracking Process**: Updates tracks based on new detections, ensuring each face maintains a unique identifier.

### Recognition

**FaceNet** generates 512-dimensional embeddings for detected faces, enabling comparison and recognition against stored embeddings in the database.

- **Embedding Generation**: Processes cropped face images to produce embeddings.
- **Similarity Calculation**: Compares embeddings using cosine similarity to identify known individuals.

### Anti-Spoofing

Implements an anti-spoofing mechanism by analyzing the variance of the Laplacian (a measure of image sharpness) to detect potential spoofed faces.

- **Process**: Converts face ROI to grayscale and computes Laplacian variance.
- **Thresholding**: Determines if the face is real based on a predefined threshold.

### Blink Detection

Monitors eye aspect ratios to detect blinks, enhancing the system's ability to analyze facial expressions and fatigue.

- **Landmark Detection**: Uses Mediapipe's Face Mesh to identify eye landmarks.
- **Aspect Ratio Calculation**: Computes the ratio to determine if eyes are closed.

### Eye Color Detection

Analyzes the iris region to determine eye color by identifying the dominant color within the detected iris landmarks.

- **Dominant Color Extraction**: Applies K-Means clustering to find the most prevalent color.
- **Color Classification**: Matches the dominant color to predefined eye color ranges.

### Face Mesh

**Mediapipe's Face Mesh** provides detailed facial landmark detection, enabling precise analysis of facial features.

- **Landmark Detection**: Identifies 468 facial landmarks for comprehensive mesh generation.
- **Visualization**: Draws tesselation, contours, and irises based on configuration settings.

### Hand Tracking

Incorporates **Mediapipe Hands** to detect and track hands within the image, adding an extra layer of analysis for gestures and interactions.

- **Detection**: Identifies hand landmarks and handedness.
- **Visualization**: Draws hand landmarks and connections with customizable colors.

### Export/Import Mechanism

Facilitates the backup and restoration of both configuration settings and the face database.

- **Combined Export**: Packages both config and database into a single pickle file for easy download.
- **Combined Import**: Allows users to upload the combined file to restore settings and data seamlessly.
- **Flexibility**: Users can choose to merge or overwrite the existing database during import.

## Packages and Credits

This project leverages several open-source packages to deliver its functionality:

- **[Gradio](https://gradio.app/)**: For building the web-based user interface.
- **[PyTorch](https://pytorch.org/)**: Backend for deep learning models.
- **[FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)**: For face embedding generation.
- **[Ultralytics YOLO](https://github.com/ultralytics/yolov5)**: For face detection.
- **[Deep SORT Realtime](https://github.com/theAIGuysCode/deep_sort_realtime)**: For face tracking.
- **[Mediapipe](https://google.github.io/mediapipe/)**: For face mesh and hand tracking.
- **[OpenCV](https://opencv.org/)**: For image processing tasks.
- **[NumPy](https://numpy.org/)**: For numerical operations.
- **[Pillow](https://python-pillow.org/)**: For image handling.
- **[Requests](https://requests.readthedocs.io/)**: For HTTP requests to download models.
- **[Logging](https://docs.python.org/3/library/logging.html)**: For logging system activities.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Feel free to contribute, report issues, or suggest features to improve this system!*
