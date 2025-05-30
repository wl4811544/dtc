"""
Setup script for Anomaly-Aware Knowledge Tracing
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anomaly-aware-kt",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Anomaly detection and handling for knowledge tracing systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/anomaly-aware-kt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0",
        "tomlkit>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "anomaly-kt-train=scripts.full_pipeline:main",
        ],
    },
)