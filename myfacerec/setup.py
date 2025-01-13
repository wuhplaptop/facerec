from setuptools import setup, find_packages

setup(
    name="myfacerec",
    version="0.1.0",
    description="Facial Recognition with YOLO + Facenet, with a default model and configurable confidence threshold.",
    author="Your Name",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
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
)
