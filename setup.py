"""
APIGen - FastAPI Backend Generator
Install: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="apigen",
    version="0.1.0",
    author="Diegoproggramer",
    author_email="",
    description="⚡ Generate complete FastAPI backends in seconds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Diegoproggramer/apigen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "sqlalchemy>=2.0.0",
        "pydantic>=2.0.0",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.5",
        "bcrypt>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "httpx>=0.24.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "apigen=apigen.cli:main",
        ],
    },
    keywords="fastapi, generator, api, backend, code-generator, crud, python",
    project_urls={
        "Bug Reports": "https://github.com/Diegoproggramer/apigen/issues",
        "Source": "https://github.com/Diegoproggramer/apigen",
    },
)
