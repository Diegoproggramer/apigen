# File: apigen/__init__.py
"""
NexaFlow APIGen — Automated REST API Code Generator
=====================================================

A production-grade code generator that transforms database schema definitions
(JSON/YAML) into complete, runnable FastAPI + SQLAlchemy 2.0 applications
with Pydantic V2 schemas, full CRUD routers, pagination, search, Alembic
migrations, and more.

Architecture overview::

    ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
    │  CLI / Entry │────▶│  APIGenerator  │────▶│ TemplateGenerator│
    │   (cli.py)   │     │ (generator.py) │     │  (templates.py)  │
    └──────────────┘     └───────┬───────┘     └──────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌───────────┐ ┌───────────┐
             │validators│ │  models   │ │ exporters │
             │  (.py)   │ │  (.py)    │ │  (.py)    │
             └──────────┘ └───────────┘ └───────────┘

Usage::

    # As a library
    from apigen import APIGenerator, GenerationConfig, SchemaDefinition
    gen = APIGenerator(config)
    gen.generate(schema)

    # From the command line
    python -m apigen --schema schema.yaml --output ./my_api --verbose

Public API:
    - APIGenerator       — Master orchestrator
    - GenerationConfig   — Generation settings model
    - SchemaDefinition   — Database schema model
    - TemplateGenerator  — Code template engine
    - ProjectExporter    — File-system writer
    - validate_full      — Schema validation entry point
"""

from __future__ import annotations

__version__: str = "1.0.0"
__author__: str = "NexaFlow Team"
__license__: str = "MIT"

# ---------------------------------------------------------------------------
# Lazy imports — keep module load time minimal
# ---------------------------------------------------------------------------

from apigen.models import (
    AuthStrategy,
    ColumnInfo,
    ColumnType,
    CRUDConfig,
    DatabaseDialect,
    DefaultValue,
    ForeignKeyInfo,
    GenerationConfig,
    IndexInfo,
    NamingConvention,
    OnDeleteAction,
    PaginationStyle,
    RelationshipInfo,
    RelationshipType,
    SchemaDefinition,
    TableInfo,
    UniqueConstraintInfo,
)
from apigen.validators import validate_full, ValidationResult
from apigen.utils import (
    Timer,
    to_snake_case,
    to_pascal_case,
    to_camel_case,
    to_plural,
    to_singular,
    safe_identifier,
    build_import_block,
    write_file,
    write_files_batch,
)
from apigen.templates import TemplateGenerator
from apigen.exporters import ProjectExporter, ExportManifest, ExportResult
from apigen.generator import APIGenerator, GenerationReport

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core orchestrator
    "APIGenerator",
    "GenerationReport",
    # Models
    "AuthStrategy",
    "ColumnInfo",
    "ColumnType",
    "CRUDConfig",
    "DatabaseDialect",
    "DefaultValue",
    "ForeignKeyInfo",
    "GenerationConfig",
    "IndexInfo",
    "NamingConvention",
    "OnDeleteAction",
    "PaginationStyle",
    "RelationshipInfo",
    "RelationshipType",
    "SchemaDefinition",
    "TableInfo",
    "UniqueConstraintInfo",
    # Validation
    "validate_full",
    "ValidationResult",
    # Templates
    "TemplateGenerator",
    # Exporters
    "ProjectExporter",
    "ExportManifest",
    "ExportResult",
    # Utilities
    "Timer",
    "to_snake_case",
    "to_pascal_case",
    "to_camel_case",
    "to_plural",
    "to_singular",
    "safe_identifier",
    "build_import_block",
    "write_file",
    "write_files_batch",
]
