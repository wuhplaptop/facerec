# setup.py

from setuptools import setup, find_packages

setup(
    name="myfacerec",
    version="0.2.0",
    description="Future-proof Facial Recognition with YOLO + Facenet, modular detectors/embedders, hooks, CLI, etc.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy",
        "torch",
        "Pillow",
        "scikit-learn",
        "ultralytics",
        "facenet-pytorch",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "myfacerec=myfacerec.cli:main", 
        ],
    },
)
