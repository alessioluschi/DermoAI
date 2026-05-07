"""Setup configuration for derm-ai-pipeline package."""
from setuptools import find_packages, setup

setup(
    name="derm-ai-pipeline",
    version="1.0.0",
    description=(
        "Dermatological AI pipeline: melanoma classification, GradCAM++, "
        "ABCDE report generation, RAG, and evaluation"
    ),
    author="DermoAI Pipeline",
    python_requires=">=3.10",
    packages=find_packages(exclude=["notebooks", "rag_kb", "data", "outputs"]),
    install_requires=[
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "grad-cam>=1.5.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.45.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "biopython>=1.81",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.0",
        "bert-score>=0.3.13",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "dermoai=src.pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
