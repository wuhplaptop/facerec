# Rolo-Rec: Modular and Extensible Facial Recognition Tool

## Overview
Rolo-Rec is a powerful, modular, and extensible facial recognition system designed for developers to customize and extend its functionalities. The tool supports face detection, embedding, and recognition with features for plugin management, combined model usage, and an extensible CLI for operations.

## Installation

Rolo-Rec can be installed directly from PyPI:
```bash
pip install rolo-rec
```
Alternatively, you can clone the repository from GitHub:
```bash
git clone https://github.com/wuhplaptop/facerec.git
cd facerec
python setup.py install
```

## Key Features
- **Extensible Plugins**: Easily add custom detectors, embedders, and data stores.
- **Combined Model Support**: Use YOLO for detection and Facenet for embeddings in a unified model.
- **Command Line Interface (CLI)**: Perform face registration, identification, model import/export, and user management.
- **Configurable Hooks**: Inject pre- and post-processing logic.

## Usage

### Command-Line Interface
Run the CLI with:
```bash
rolo-rec <command> [options]
```

### Available Commands

#### 1. Register a User
Registers a user by processing one or more images.
```bash
rolo-rec register --user <USER_ID> --images <IMAGE_PATHS> [--conf <CONFIDENCE>] [--model-path <MODEL_PATH>] [--detector <DETECTOR_PLUGIN>] [--embedder <EMBEDDER_PLUGIN>] [--combined-model-path <PATH>]
```
Example:
```bash
rolo-rec register --user alice --images alice1.jpg alice2.jpg
```

#### 2. Identify Faces
Identifies faces in an input image.
```bash
rolo-rec identify --image <IMAGE_PATH> [--threshold <RECOGNITION_THRESHOLD>] [--conf <CONFIDENCE>] [--model-path <MODEL_PATH>] [--detector <DETECTOR_PLUGIN>] [--embedder <EMBEDDER_PLUGIN>] [--combined-model-path <PATH>]
```
Example:
```bash
rolo-rec identify --image group_photo.jpg --threshold 0.7
```

#### 3. Export Data
Exports registered user data to a JSON file.
```bash
rolo-rec export-data --output <OUTPUT_PATH>
```
Example:
```bash
rolo-rec export-data --output user_data.json
```

#### 4. Import Data
Imports user data from a JSON file.
```bash
rolo-rec import-data --file <JSON_PATH>
```
Example:
```bash
rolo-rec import-data --file user_data.json
```

#### 5. List Users
Lists all registered users.
```bash
rolo-rec list-users
```
Example output:
```
Registered Users:
- alice (5 embeddings)
- bob (3 embeddings)
```

#### 6. Delete a User
Deletes a registered user.
```bash
rolo-rec delete-user --user <USER_ID>
```
Example:
```bash
rolo-rec delete-user --user alice
```

#### 7. Export Model
Exports the combined model and user data to a `.pt` file.
```bash
rolo-rec export-model --export-path <OUTPUT_PATH>
```
Example:
```bash
rolo-rec export-model --export-path combined_model.pt
```

#### 8. Import Model
Imports a combined model from a `.pt` file.
```bash
rolo-rec import-model --import-path <MODEL_PATH>
```
Example:
```bash
rolo-rec import-model --import-path combined_model.pt
```

### Example Usage

#### Programmatic Interface
Use Rolo-Rec in your Python projects:

```python
from rolo_rec import FacialRecognition, Config

# Configuration
config = Config()
facial_recognition = FacialRecognition(config)

# Register user
images = ["path/to/image1.jpg", "path/to/image2.jpg"]
response = facial_recognition.register_user("alice", images)
print(response)

# Identify user
from PIL import Image
image = Image.open("path/to/input.jpg")
results = facial_recognition.identify_user(image, threshold=0.6)
for result in results:
    print(result)
```

## How It Works

### Architecture
Rolo-Rec is built around modular components:

1. **Detectors**: Locate faces in an image.
   - Default: YOLO-based `YOLOFaceDetector`
2. **Embedders**: Extract facial embeddings.
   - Default: Facenet-based `FacenetEmbedder`
3. **Data Store**: Store and retrieve user data.
   - Default: JSON-based `JSONUserDataStore`
4. **Hooks**: Pre- and post-processing hooks for detection and embedding.
5. **Plugins**: Add custom detectors, embedders, and data stores.

### Combined Model
- Combines YOLO and Facenet into a single model.
- Supports seamless face detection and embedding extraction.

### Plugins
Plugins allow developers to extend Rolo-Rec. Add custom plugins by defining them in `setup.py` under `entry_points`:

```python
setup(
    ...
    entry_points={
        'rolo_rec.detectors': [
            'custom_detector = mymodule:CustomDetector',
        ],
        'rolo_rec.embedders': [
            'custom_embedder = mymodule:CustomEmbedder',
        ],
        'rolo_rec.data_stores': [
            'custom_data_store = mymodule:CustomDataStore',
        ],
    },
)
```

## Advanced Features

### Configurations
Configure various aspects of the tool through the `Config` class:
- `yolo_model_path`: Path to the YOLO model.
- `conf_threshold`: Confidence threshold for detection.
- `device`: Compute device (CPU/GPU).
- `user_data_path`: Path for saving user data.

### Hooks
Inject custom logic with hooks:
```python
from rolo_rec.hooks import Hooks
hooks = Hooks()

# Register a hook
hooks.register_before_detect(lambda image: preprocess(image))
```

### Model Export/Import
Save or load the combined YOLO and Facenet model:
```python
model.save_model("path/to/model.pt")
loaded_model = CombinedFacialRecognitionModel.load_model("path/to/model.pt")
```

## Contributing
We welcome contributions! Please fork the repo and submit a pull request.


