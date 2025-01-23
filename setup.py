# setup.py

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rolo-rec",  # Renamed to "rolo-rec" for PyPI
    version="0.5.0",
    description="Future-proof Facial Recognition with YOLO + Facenet, modular detectors/embedders, hooks, CLI, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="wuhp",
    author_email="your-email@example.com",
    url="https://github.com/wuhplaptop/facerec",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'myfacerec': ['models/*.pt'],
    },
    install_requires=[
        "requests",
        "numpy",
        "torch",
        "Pillow",
        "scikit-learn",
        "ultralytics",
        "facenet-pytorch",
        "opencv-python",  # For pose estimation
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "rolo-rec=myfacerec.cli:main",
        ],
        "myfacerec.detectors": [
            # "yolo_face=myfacerec.detectors:YOLOFaceDetector",
        ],
        "myfacerec.embedders": [
            # "facenet=myfacerec.embedders:FacenetEmbedder",
        ],
        "myfacerec.data_stores": [
            # "json_store=myfacerec.data_store:JSONUserDataStore",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    keywords="facial recognition yolo facenet",
    project_urls={
        "Bug Tracker": "https://github.com/wuhplaptop/facerec/issues",
        "Documentation": "https://github.com/wuhplaptop/facerec#readme",
        "Source Code": "https://github.com/wuhplaptop/facerec",
    },
)
