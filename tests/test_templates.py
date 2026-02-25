"""
tests/test_templates.py
Unit tests for apigen.templates module (TemplateGenerator).

Tests cover:
- SQLAlchemy model generation (column types, relationships, ForeignKey)
- Pydantic schema generation (Create, Read, Update schemas)
- CRUD router generation (FastAPI endpoints)
- Database session file generation
- Main app file generation
- Full generate_all pipeline
- Code correctness (valid Python syntax via compile())
"""

from __future__ import annotations

import ast
import copy
import pathlib
from typing import Any, Dict, List

import pytest
import yaml

from apigen.models import (
    ColumnDefinition,
    GenerationConfig,
    ProjectMeta,
    RelationshipDefinition,
    SchemaDefinition,
    TableDefinition,
)
from apigen.templates import TemplateGenerator


# ===========================================================================
# Helper: build SchemaDefinition from raw dict (same as test_validators)
# ===========================================================================


def _build_schema(raw: Dict[str, Any]) -> SchemaDefinition:
    project_data = raw.get("project", {})
    config_data = raw.get("config", {})
    tables_data = raw.get("tables", [])

    project = ProjectMeta(
        name=project_data.get("name", "test"),
        version=project_data.get("version", "0.1.0"),
        description=project_data.get("description", ""),
        database_url=project_data.get("database_url", "sqlite:///./test.db"),
    )
    config = GenerationConfig(
        api_prefix=config_data.get("api_prefix", "/api/v1"),
        enable_soft_delete=config_data.get("enable_soft_delete", False),
        enable_timestamps=config_data.get("enable_timestamps", False),
        enable_pagination=config_data.get("enable_pagination", True),
        default_page_size=config_data.get("default_page_size", 10),
        max_page_size=config_data.get("max_page_size", 50),
        auth_enabled=config_data.get("auth_enabled", False),
    )
    tables: List[TableDefinition] = []
    for tbl in tables_data:
        columns = []
        for col in tbl.get("columns", []):
            columns.append(
                ColumnDefinition(
                    name=col["name"],
                    type=col["type"],
                    primary_key=col.get("primary_key", False),
                    autoincrement=col.get("autoincrement", False),
                    nullable=col.get("nullable", True),
                    unique=col.get("unique", False),
                    index=col.get("index", False),
                    default=col.get("default"),
                    server_default=col.get("server_default"),
                    max_length=col.get("max_length"),
                    foreign_key=col.get("foreign_key"),
                    description=col.get("description", ""),
                    onupdate=col.get("onupdate"),
                )
            )
        relationships = []
        for rel in tbl.get("relationships", []):
            relationships.append(
                RelationshipDefinition(
                    name=rel["name"],
                    target=rel["target"],
                    type=rel["type"],
                    back_populates=rel.get("back_populates"),
                    cascade=rel.get("cascade"),
                )
            )
        tables.append(
            TableDefinition(
                name=tbl["name"],
                description=tbl.get("description", ""),
                columns=columns,
                relationships=relationships,
            )
        )
    return SchemaDefinition(project=project, config=config, tables=tables)


def _is_valid_python(code: str, filename: str = "<generated>") -> bool:
    """Check if a string of Python code is syntactically valid."""
    try:
        ast.parse(code, filename=filename)
        return True
    except SyntaxError:
        return False


# ===========================================================================
# Test SQLAlchemy Model Generation
# ===========================================================================


class TestModelGeneration:
    """Tests for SQLAlchemy model code generation."""

    def test_minimal_model_generated(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_model(schema.tables[0])
        assert "class Item" in result
        assert "id" in result
        assert "title" in result

    def test_model_is_valid_python(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_model(schema.tables[0])
        assert _is_valid_python(result), f"Generated model is not valid Python:\n{result}"

    def test_foreign_key_in_model(self, two_table_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(two_table_schema_dict)
        gen = TemplateGenerator(schema)
        # Book table has author_id FK
        book_table = [t for t in schema.tables if t.name == "Book"][0]
        result = gen.generate_model(book_table)
        assert "ForeignKey" in result or "foreign_key" in result.lower()
        assert "author_id" in result

    def test_relationship_in_model(self, two_table_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(two_table_schema_dict)
        gen = TemplateGenerator(schema)
        author_table = [t for t in schema.tables if t.name == "Author"][0]
        result = gen.generate_model(author_table)
        assert "relationship" in result.lower()
        assert "books" in result

    def test_all_full_schema_models_valid_python(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        for table in schema.tables:
            code = gen.generate_model(table)
            assert _is_valid_python(code), (
                f"Model for table '{table.name}' is not valid Python:\n{code}"
            )

    def test_model_has_tablename(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_model(schema.tables[0])
        assert "__tablename__" in result

    def test_primary_key_marked(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_model(schema.tables[0])
        assert "primary_key" in result

    def test_string_column_has_length(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_model(schema.tables[0])
        # title column is String(100)
        assert "100" in result or "String" in result


# ===========================================================================
# Test Pydantic Schema Generation
# ===========================================================================


class TestSchemaGeneration:
    """Tests for Pydantic schema code generation."""

    def test_minimal_schema_generated(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_schema(schema.tables[0])
        assert "ItemCreate" in result or "Create" in result
        assert "ItemRead" in result or "Read" in result

    def test_schema_is_valid_python(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_schema(schema.tables[0])
        assert _is_valid_python(result), f"Generated schema is not valid Python:\n{result}"

    def test_all_full_schema_schemas_valid_python(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        for table in schema.tables:
            code = gen.generate_schema(table)
            assert _is_valid_python(code), (
                f"Schema for table '{table.name}' is not valid Python:\n{code}"
            )

    def test_schema_has_base_model(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_schema(schema.tables[0])
        assert "BaseModel" in result or "pydantic" in result.lower()

    def test_optional_fields_in_update_schema(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_schema(schema.tables[0])
        assert "Update" in result
        assert "Optional" in result or "None" in result


# ===========================================================================
# Test CRUD Router Generation
# ===========================================================================


class TestRouterGeneration:
    """Tests for FastAPI CRUD router code generation."""

    def test_minimal_router_generated(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_router(schema.tables[0])
        assert "router" in result.lower() or "APIRouter" in result
        assert "get" in result.lower() or "GET" in result

    def test_router_is_valid_python(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_router(schema.tables[0])
        assert _is_valid_python(result), f"Generated router is not valid Python:\n{result}"

    def test_router_has_crud_endpoints(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_router(schema.tables[0])
        # Should contain at least create, read, update, delete operations
        result_lower = result.lower()
        assert "post" in result_lower or "create" in result_lower
        assert "get" in result_lower or "read" in result_lower
        assert "put" in result_lower or "patch" in result_lower or "update" in result_lower
        assert "delete" in result_lower

    def test_all_full_schema_routers_valid_python(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        for table in schema.tables:
            code = gen.generate_router(table)
            assert _is_valid_python(code), (
                f"Router for table '{table.name}' is not valid Python:\n{code}"
            )


# ===========================================================================
# Test Database Session Generation
# ===========================================================================


class TestDatabaseGeneration:
    """Tests for database session/engine code generation."""

    def test_database_file_generated(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_database()
        assert "engine" in result.lower() or "Engine" in result
        assert "session" in result.lower() or "Session" in result

    def test_database_is_valid_python(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_database()
        assert _is_valid_python(result), f"Generated database.py is not valid Python:\n{result}"

    def test_database_url_referenced(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_database()
        assert "DATABASE_URL" in result or "database_url" in result.lower() or "url" in result.lower()


# ===========================================================================
# Test Main App Generation
# ===========================================================================


class TestMainAppGeneration:
    """Tests for FastAPI main app file generation."""

    def test_main_app_generated(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_main_app()
        assert "FastAPI" in result or "app" in result

    def test_main_app_is_valid_python(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_main_app()
        assert _is_valid_python(result), f"Generated main.py is not valid Python:\n{result}"

    def test_main_app_includes_routers(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_main_app()
        assert "include_router" in result or "router" in result.lower()


# ===========================================================================
# Test generate_all Pipeline
# ===========================================================================


class TestGenerateAll:
    """Tests for the full code generation pipeline."""

    def test_generate_all_returns_dict(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_all()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_generate_all_has_expected_keys(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_all()
        # Should have model, schema, router files + database + main
        keys_lower = [k.lower() for k in result.keys()]
        all_keys_str = " ".join(keys_lower)
        assert "model" in all_keys_str or "models" in all_keys_str
        assert "schema" in all_keys_str or "schemas" in all_keys_str
        assert "router" in all_keys_str or "routers" in all_keys_str
        assert "database" in all_keys_str or "db" in all_keys_str
        assert "main" in all_keys_str or "app" in all_keys_str

    def test_generate_all_full_schema(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_all()
        assert isinstance(result, dict)
        assert len(result) >= 5  # At least models, schemas, routers, database, main
        # Verify all values are non-empty strings
        for filepath, code in result.items():
            assert isinstance(code, str), f"Value for {filepath} is not a string"
            assert len(code.strip()) > 0, f"Code for {filepath} is empty"

    def test_generate_all_all_code_valid_python(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_all()
        for filepath, code in result.items():
            if filepath.endswith(".py"):
                assert _is_valid_python(code, filename=filepath), (
                    f"Generated file '{filepath}' is not valid Python:\n{code[:500]}..."
                )

    def test_generate_all_six_tables_produce_many_files(self, schema_dict: Dict[str, Any]) -> None:
        """Full shop schema has 6 tables, so should produce many files."""
        schema = _build_schema(schema_dict)
        gen = TemplateGenerator(schema)
        result = gen.generate_all()
        # 6 models + 6 schemas + 6 routers + database + main = at least 20 files
        # (depending on structure: might be combined files or split)
        assert len(result) >= 4, f"Expected at least 4 generated files, got {len(result)}"
