# setup.py

from setuptools import setup, find_packages

setup(
    name='face_pipeline',  # Ensure this matches your intended package name
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive face recognition and analysis pipeline with sharing capabilities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/face_pipeline',  # Update with your repository URL
    packages=find_packages(),  # This should automatically find 'face_pipeline' package
    include_package_data=True,  # Ensures non-code files are included as per MANIFEST.in
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'torchvision',
        'facenet-pytorch',
        'ultralytics',
        'deep_sort_realtime',
        'mediapipe',
        'Pillow',
        'streamlit',
        'streamlit-webrtc',
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',  # Update as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'face-pipeline=face_pipeline.core:main',
        ],
    },
)
