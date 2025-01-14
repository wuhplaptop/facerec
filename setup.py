# setup.py

from setuptools import setup, find_packages

setup(
    name="rolo-rec",
    version="0.4.0",  # Incremented version
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
        # Removed "pkg_resources" as it is part of setuptools
    ],
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
