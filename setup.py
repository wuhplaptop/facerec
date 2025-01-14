# setup.py

from setuptools import setup, find_packages

setup(
    name="rolo-rec",
    version="0.4.7",
    description="Future-proof Facial Recognition with YOLO + Facenet, modular detectors/embedders, hooks, CLI, etc.",
    author="wuhp",
    packages=find_packages(),
    include_package_data=True,  # Ensures package data from MANIFEST.in is included
    install_requires=[
        "requests",
        "numpy",
        "torch",
        "Pillow",
        "scikit-learn",
        "ultralytics",
        "facenet-pytorch",
        "torchvision<0.18.0,>=0.17.0",
        "tqdm<5.0.0,>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pluggy<2,>=1.5",
            "iniconfig",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "rolo-rec=myfacerec.cli:main",
        ],
        "rolo_rec.detectors": [
            # Register custom detectors here
            # Example:
            # "yolo_detector=myfacerec.plugins.sample_plugin:SampleDetector",
        ],
        "rolo_rec.embedders": [
            "sample_embedder=myfacerec.plugins.sample_plugin:SampleEmbedder",
        ],
        "rolo_rec.data_stores": [
            # Register custom data stores here
            # Example:
            # "json_data_store=myfacerec.plugins.sample_plugin:SampleDataStore",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
