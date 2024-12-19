"""Setup file for report_gen_eval package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="report_gen_eval",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for evaluating AI-generated reports",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/report_gen_eval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "together>=0.1.0",
        "tqdm>=4.65.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "report-eval=report_gen_eval.cli:main",
        ],
    },
) 