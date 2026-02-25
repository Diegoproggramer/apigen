"""
tests/conftest.py
Shared fixtures for the apigen test suite.

All fixtures are session-scoped or function-scoped as appropriate.
No external mocking libraries are used; real file I/O is performed
inside temporary directories managed by pytest's tmp_path fixtures.
"""

from __future__ import annotations

import copy
import os
import pathlib
import tempfile
import textwrap
from typing import Any, Dict, List

import pytest
import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_EXAMPLE_PATH: pathlib.Path = ROOT_DIR / "schema_example.yaml"


# ---------------------------------------------------------------------------
# Raw schema data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def raw_schema_dict() -> Dict[str, Any]:
    """Load the reference schema_example.yaml once per session and return as dict."""
    assert SCHEMA_EXAMPLE_PATH.exists(), (
        f"Reference schema not found at {SCHEMA_EXAMPLE_PATH}. "
        "Make sure schema_example.yaml is in the project root."
    )
    with open(SCHEMA_EXAMPLE_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "Top-level YAML must be a mapping."
    return data


@pytest.fixture()
def schema_dict(raw_schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy so each test can mutate freely."""
    return copy.deepcopy(raw_schema_dict)


@pytest.fixture()
def schema_yaml_path(schema_dict: Dict[str, Any], tmp_path: pathlib.Path) -> pathlib.Path:
    """Write the schema dict to a temporary YAML file and return its path."""
    path = tmp_path / "schema.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(schema_dict, fh, default_flow_style=False, allow_unicode=True)
    return path


# ---------------------------------------------------------------------------
# Minimal / edge-case schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_schema_dict() -> Dict[str, Any]:
    """Smallest valid schema: one table, one primary-key column, no relationships."""
    return {
        "project": {
            "name": "minimal_app",
            "version": "0.1.0",
            "description": "Minimal test project",
            "database_url": "sqlite+aiosqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "Item",
                "description": "A simple item",
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "primary_key": True,
                        "autoincrement": True,
                    },
                    {
                        "name": "title",
                        "type": "string",
                        "max_length": 100,
                        "nullable": False,
                    },
                ],
                "relationships": [],
            }
        ],
    }


@pytest.fixture()
def minimal_schema_yaml_path(
    minimal_schema_dict: Dict[str, Any], tmp_path: pathlib.Path
) -> pathlib.Path:
    """Write minimal schema to a temp YAML and return the path."""
    path = tmp_path / "minimal_schema.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(minimal_schema_dict, fh, default_flow_style=False)
    return path


@pytest.fixture()
def two_table_schema_dict() -> Dict[str, Any]:
    """Two tables with a foreign key relationship for topological sort testing."""
    return {
        "project": {
            "name": "two_table_app",
            "version": "0.1.0",
            "description": "Two-table project for relationship tests",
            "database_url": "sqlite+aiosqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": True,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "Author",
                "description": "Book authors",
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "primary_key": True,
                        "autoincrement": True,
                    },
                    {
                        "name": "name",
                        "type": "string",
                        "max_length": 150,
                        "nullable": False,
                    },
                ],
                "relationships": [
                    {
                        "name": "books",
                        "target": "Book",
                        "type": "one-to-many",
                        "back_populates": "author",
                        "cascade": "all, delete-orphan",
                    }
                ],
            },
            {
                "name": "Book",
                "description": "Books in the library",
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "primary_key": True,
                        "autoincrement": True,
                    },
                    {
                        "name": "title",
                        "type": "string",
                        "max_length": 250,
                        "nullable": False,
                    },
                    {
                        "name": "author_id",
                        "type": "integer",
                        "nullable": False,
                        "foreign_key": "Author.id",
                        "index": True,
                    },
                ],
                "relationships": [
                    {
                        "name": "author",
                        "target": "Author",
                        "type": "many-to-one",
                        "back_populates": "books",
                    }
                ],
            },
        ],
    }


@pytest.fixture()
def two_table_yaml_path(
    two_table_schema_dict: Dict[str, Any], tmp_path: pathlib.Path
) -> pathlib.Path:
    """Write the two-table schema to a temp file."""
    path = tmp_path / "two_table_schema.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(two_table_schema_dict, fh, default_flow_style=False)
    return path


# ---------------------------------------------------------------------------
# Invalid schema fixtures (for negative testing)
# ---------------------------------------------------------------------------


@pytest.fixture()
def schema_missing_project() -> Dict[str, Any]:
    """Schema with 'project' section entirely missing."""
    return {
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "Dummy",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True}
                ],
            }
        ],
    }


@pytest.fixture()
def schema_no_primary_key() -> Dict[str, Any]:
    """Schema where a table has no primary key column."""
    return {
        "project": {
            "name": "bad_pk",
            "version": "0.1.0",
            "description": "Missing PK test",
            "database_url": "sqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "NoPK",
                "description": "Table without a primary key",
                "columns": [
                    {"name": "value", "type": "string", "max_length": 50, "nullable": False}
                ],
                "relationships": [],
            }
        ],
    }


@pytest.fixture()
def schema_duplicate_table_names() -> Dict[str, Any]:
    """Schema with two tables having the same name."""
    base_table = {
        "name": "Conflict",
        "description": "Duplicate name",
        "columns": [
            {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True}
        ],
        "relationships": [],
    }
    return {
        "project": {
            "name": "dup_tables",
            "version": "0.1.0",
            "description": "Duplicate table name test",
            "database_url": "sqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [copy.deepcopy(base_table), copy.deepcopy(base_table)],
    }


@pytest.fixture()
def schema_broken_foreign_key() -> Dict[str, Any]:
    """Schema where a foreign key references a non-existent table."""
    return {
        "project": {
            "name": "broken_fk",
            "version": "0.1.0",
            "description": "Broken FK test",
            "database_url": "sqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "Orphan",
                "description": "Has FK to non-existent table",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                    {
                        "name": "ghost_id",
                        "type": "integer",
                        "nullable": False,
                        "foreign_key": "GhostTable.id",
                    },
                ],
                "relationships": [],
            }
        ],
    }


@pytest.fixture()
def schema_invalid_column_type() -> Dict[str, Any]:
    """Schema with an invalid/unsupported column type."""
    return {
        "project": {
            "name": "bad_type",
            "version": "0.1.0",
            "description": "Invalid column type test",
            "database_url": "sqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "BadType",
                "description": "Has unsupported column type",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                    {"name": "data", "type": "hyperblob", "nullable": True},
                ],
                "relationships": [],
            }
        ],
    }


@pytest.fixture()
def schema_circular_foreign_keys() -> Dict[str, Any]:
    """Schema with circular foreign key dependencies (A -> B -> A)."""
    return {
        "project": {
            "name": "circular_fk",
            "version": "0.1.0",
            "description": "Circular FK test",
            "database_url": "sqlite:///./test.db",
        },
        "config": {
            "api_prefix": "/api/v1",
            "enable_soft_delete": False,
            "enable_timestamps": False,
            "enable_pagination": True,
            "default_page_size": 10,
            "max_page_size": 50,
            "auth_enabled": False,
        },
        "tables": [
            {
                "name": "Alpha",
                "description": "References Beta",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                    {"name": "beta_id", "type": "integer", "nullable": True, "foreign_key": "Beta.id"},
                ],
                "relationships": [],
            },
            {
                "name": "Beta",
                "description": "References Alpha",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                    {"name": "alpha_id", "type": "integer", "nullable": True, "foreign_key": "Alpha.id"},
                ],
                "relationships": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Output directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide a clean output directory inside tmp_path."""
    out = tmp_path / "generated_output"
    out.mkdir(parents=True, exist_ok=True)
    return out
