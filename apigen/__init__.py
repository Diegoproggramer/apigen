"""
APIGen - AI-Powered API Generator
Automatically generate production-ready FastAPI backends
"""

__version__ = "0.1.0"
__author__ = "Diegoproggramer"

from apigen.generator import APIGenerator
from apigen.database import DatabaseModeler
from apigen.auth import AuthGenerator
from apigen.router import RouterGenerator
from apigen.schemas import SchemaGenerator

__all__ = [
    "APIGenerator",
    "DatabaseModeler", 
    "AuthGenerator",
    "RouterGenerator",
    "SchemaGenerator",
]

print(f"⚡ APIGen v{__version__} loaded - AI-Powered API Generator")
