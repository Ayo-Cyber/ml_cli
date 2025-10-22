from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define dependencies inline (not from requirements.txt for build isolation)
required = [
    "click>=8.0.0",
    "rich-click>=1.6.0",
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "tpot>=0.12.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "fastapi>=0.68.0",
    "uvicorn>=0.15.0",
    "pyyaml>=5.4.0",
    "joblib>=1.0.0",
    "ydata-profiling>=4.0.0",
    "questionary>=1.10.0",
    "pydantic>=2.0.0",
    "requests>=2.26.0",
]

setup(
    name="ml-cli",
    version="1.0.0",
    author="Atunrase Ayomide",
    author_email="atunraseayomide@gmail.com",
    description="A comprehensive CLI tool for end-to-end machine learning workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ayo-Cyber/ml_cli",
    project_urls={
        "Bug Tracker": "https://github.com/Ayo-Cyber/ml_cli/issues",
        "Documentation": "https://github.com/Ayo-Cyber/ml_cli#readme",
        "Source Code": "https://github.com/Ayo-Cyber/ml_cli",
        "Changelog": "https://github.com/Ayo-Cyber/ml_cli/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=required,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml=ml_cli.cli:cli",
        ],
    },
    include_package_data=True,
    keywords=[
        "machine-learning",
        "ml",
        "cli",
        "automl",
        "data-science",
        "tpot",
        "automation",
        "preprocessing",
        "eda",
        "model-training",
    ],
)
