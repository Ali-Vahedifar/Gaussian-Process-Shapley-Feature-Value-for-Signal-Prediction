"""
Setup script for GP+SFV Tactile Internet framework
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shapley-gp-ti",
    version="1.0.0",
    author="Mohammad Ali Vahedifar, Arthur, Qi Zhang",
    author_email="av@ece.au.dk, qz@ece.au.dk",
    description="Shapley Features for Robust Signal Prediction in Tactile Internet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gp-sfv-train=train:main",
            "gp-sfv-evaluate=evaluate:main",
        ],
    },
    keywords=[
        "tactile internet",
        "gaussian process",
        "shapley values",
        "signal prediction",
        "haptic communication",
        "machine learning",
        "deep learning",
        "feature selection",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction/issues",
        "Source": "https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction",
        "Paper": "https://arxiv.org/abs/xxxx.xxxxx",  # Update with actual arXiv link
        "TOAST Project": "https://toast-doctoral-network.eu/",
    },
)
