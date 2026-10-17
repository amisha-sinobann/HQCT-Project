from setuptools import setup, find_packages

setup(
    name="hqct",
    version="0.1.0",
    description="Hybrid Quantum-Classical Transformer for NLP",
    author="Amisha Singh (implementation)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "pennylane>=0.34.0",
        "pennylane-lightning>=0.34.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "noise": ["qiskit>=0.45.0", "qiskit-aer>=0.13.0"],
    },
)
