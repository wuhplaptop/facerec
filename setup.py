# setup.py

from setuptools import setup, find_packages

setup(
    name='face_pipeline',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive face recognition and analysis pipeline with sharing capabilities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/face_pipeline',
    packages=find_packages(),
    include_package_data=True,
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
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'face-pipeline=face_pipeline.core:main',
        ],
    },
)
