# setup.py

from setuptools import setup, find_packages
import os

# Read the README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rolo-rec",  # Renamed to "rolo-rec" for PyPI
    version="0.2.0",
    description="Future-proof Facial Recognition with YOLO + Facenet, modular detectors/embedders, hooks, CLI, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="wuhp",
    author_email="your-email@example.com",  # Replace with your actual email
    url="https://github.com/wuhplaptop/facerec",  # Replace with your actual repository URL
    packages=find_packages(),
    include_package_data=True,  # Include package data as specified in MANIFEST.in
    package_data={
        'myfacerec': ['models/*.pt'],  # Include all .pt files in myfacerec/models/
    },
    install_requires=[
        "requests",
        "numpy",
        "torch",
        "Pillow",
        "scikit-learn",
        "ultralytics",
        "facenet-pytorch",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            # Add other development/testing dependencies here
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your actual license if different
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            # This means users can now run `rolo-rec` from the command line
            "rolo-rec=myfacerec.cli:main", 
        ],
    },
    license="MIT",  # Replace with your actual license if different
    keywords="facial recognition yolo facenet",
    project_urls={
        "Bug Tracker": "https://github.com/wuhplaptop/facerec/issues",
        "Documentation": "https://github.com/wuhplaptop/facerec#readme",
        "Source Code": "https://github.com/wuhplaptop/facerec",
    },
)
