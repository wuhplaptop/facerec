from setuptools import setup

setup(
    name="face-pipeline",  # PyPI package name
    version="0.0.1",       # Increment to release new versions
    py_modules=["face_pipeline"],  # Because your code is in face_pipeline.py
    install_requires=[
        "requests",
        "numpy",
        "opencv-python",
        "torch",
        "Pillow",
        "gradio",
        "ultralytics",
        "facenet-pytorch",
        "torchvision",
        "deep_sort_realtime",
        "mediapipe",
    ],
    entry_points={
        "console_scripts": [
            # This creates a CLI command 'face-pipeline' that calls main() in face_pipeline.py
            "face-pipeline = face_pipeline:main"
        ]
    },
    author="YourName",
    author_email="you@example.com",
    description="Single-file face pipeline with YOLO, Mediapipe, and Gradio",
    url="https://github.com/yourusername/yourrepo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
