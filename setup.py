"""
Setup script for hetero_framework package.
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hetero_framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A distributed deep learning framework for heterogeneous GPU training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hetrogpu",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "viz": [
            "graphviz>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hetero-worker=hetero_framework.trainer.worker:main",
        ],
    },
)

