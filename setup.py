#!/usr/bin/env python3
"""
Setup script for Audio Processing Application
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-processing-app",
    version="1.0.0",
    author="Audio Processing Team",
    author_email="audio@example.com",
    description="Ứng dụng xử lý âm thanh với Equalizer, Giảm nhiễu ML, và Phân loại thể loại nhạc",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/audio-processing-app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gui": [
            "tkinter-tooltip>=2.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio-processor=main:main",
            "audio-train=main:train_models",
            "audio-demo=demo:main",
            "audio-test=test_audio:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="audio processing machine learning equalizer noise reduction genre classification",
    project_urls={
        "Bug Reports": "https://github.com/example/audio-processing-app/issues",
        "Source": "https://github.com/example/audio-processing-app",
        "Documentation": "https://github.com/example/audio-processing-app/blob/main/README.md",
    },
) 