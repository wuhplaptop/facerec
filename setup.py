# setup.py

from setuptools import setup, find_packages

setup(
    name="rolo-rec",
    version="0.4.5",
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
            # Add custom detectors here or in plugins
        ],
        "rolo_rec.embedders": [
            # Add custom embedders here or in plugins
        ],
        "rolo_rec.data_stores": [
            # Add custom data stores here or in plugins
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
