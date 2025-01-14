# setup.py

from setuptools import setup, find_packages

setup(
    name="rolo-rec",  # Renamed to "rolo-rec" for PyPI
    version="0.3.0",
    description="Future-proof Facial Recognition with YOLO + Facenet, modular detectors/embedders, hooks, CLI, etc.",
    author="wuhp",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy",
        "torch",
        "Pillow",
        "scikit-learn",
        "ultralytics",
        "facenet-pytorch",
        "pkg_resources",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            # This means users can now run `rolo-rec` from the command line
            "rolo-rec=myfacerec.cli:main", 
        ],
        "rolo_rec.detectors": [
            # Example: 'yolo = myfacerec.detectors:YOLOFaceDetector',
            # Add custom detectors here or in plugins
        ],
        "rolo_rec.embedders": [
            # Example: 'facenet = myfacerec.embedders:FacenetEmbedder',
            # Add custom embedders here or in plugins
        ],
        "rolo_rec.data_stores": [
            # Example: 'json = myfacerec.data_store:JSONUserDataStore',
            # Add custom data stores here or in plugins
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
