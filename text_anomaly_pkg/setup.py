from setuptools import setup, find_packages

setup(
    name="text_anomaly_pkg",
    version="1.0.0",
    description="Dueling Double-DQN Text Anomaly Detection with PER Pipeline",
    author="AI Collaborator",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.8",
)
