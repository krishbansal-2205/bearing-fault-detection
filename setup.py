# setup.py
"""
Setup script for bearing-fault-detection package.
"""

from setuptools import setup, find_packages

setup(
    name="bearing-fault-detection",
    version="1.0.0",
    description="Bearing Fault Detection using 1D-CNN with Time-Based Split",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
