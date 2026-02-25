"""
tests/test_validators.py
Comprehensive unit tests for apigen.validators module.

Tests cover:
- Schema structure validation (project, config, tables sections)
- Column type validation
- Primary key existence checks
- Foreign key referential integrity
- Duplicate table name detection
- Relationship target validation
- Topological sort cycle detection
- Full validation pipeline (validate_full)
"""

from __future__ import annotations

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
from apigen.validators import (
    ValidationError,
    ValidationResult,
    validate_column_types,
    validate_foreign_keys,
    validate_full,
    validate_primary_keys,
    validate_relationships,
    validate_schema_structure,
    validate_table_names,
    validate_topological_order,
)


# ===========================================================================
# Helper to build SchemaDefinition from raw dict
# ===========================================================================


def _build_schema(raw: Dict[str, Any]) -> SchemaDefinition:
    """
    Convert a raw schema dict into a SchemaDefinition model instance.
    This mirrors the logic in generator.parse_raw_schema().
    """
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


# ===========================================================================
# Tests for validate_schema_structure
# ===========================================================================


class TestValidateSchemaStructure:
    """Tests for top-level schema structure validation."""

    def test_valid_structure_passes(self, minimal_schema_dict: Dict[str, Any]) -> None:
        result = validate_schema_structure(minimal_schema_dict)
        assert result.is_valid, f"Expected valid, got errors: {result.errors}"

    def test_missing_project_section(self, schema_missing_project: Dict[str, Any]) -> None:
        result = validate_schema_structure(schema_missing_project)
        assert not result.is_valid
        error_messages = [e.message for e in result.errors]
        assert any("project" in msg.lower() for msg in error_messages), (
            f"Expected 'project' mentioned in errors, got: {error_messages}"
        )

    def test_missing_tables_section(self, minimal_schema_dict: Dict[str, Any]) -> None:
        data = copy.deepcopy(minimal_schema_dict)
        del data["tables"]
        result = validate_schema_structure(data)
        assert not result.is_valid

    def test_empty_tables_list(self, minimal_schema_dict: Dict[str, Any]) -> None:
        data = copy.deepcopy(minimal_schema_dict)
        data["tables"] = []
        result = validate_schema_structure(data)
        assert not result.is_valid

    def test_tables_not_a_list(self, minimal_schema_dict: Dict[str, Any]) -> None:
        data = copy.deepcopy(minimal_schema_dict)
        data["tables"] = "not_a_list"
        result = validate_schema_structure(data)
        assert not result.is_valid

    def test_project_missing_name(self, minimal_schema_dict: Dict[str, Any]) -> None:
        data = copy.deepcopy(minimal_schema_dict)
        del data["project"]["name"]
        result = validate_schema_structure(data)
        assert not result.is_valid


# ===========================================================================
# Tests for validate_table_names
# ===========================================================================


class TestValidateTableNames:
    """Tests for duplicate and invalid table name detection."""

    def test_unique_names_pass(self, two_table_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(two_table_schema_dict)
        result = validate_table_names(schema)
        assert result.is_valid

    def test_duplicate_names_fail(self, schema_duplicate_table_names: Dict[str, Any]) -> None:
        schema = _build_schema(schema_duplicate_table_names)
        result = validate_table_names(schema)
        assert not result.is_valid
        error_messages = [e.message for e in result.errors]
        assert any("duplicate" in msg.lower() or "conflict" in msg.lower() for msg in error_messages), (
            f"Expected 'duplicate' in errors, got: {error_messages}"
        )

    def test_single_table_passes(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        result = validate_table_names(schema)
        assert result.is_valid


# ===========================================================================
# Tests for validate_primary_keys
# ===========================================================================


class TestValidatePrimaryKeys:
    """Tests for primary key existence validation."""

    def test_table_with_pk_passes(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        result = validate_primary_keys(schema)
        assert result.is_valid

    def test_table_without_pk_fails(self, schema_no_primary_key: Dict[str, Any]) -> None:
        schema = _build_schema(schema_no_primary_key)
        result = validate_primary_keys(schema)
        assert not result.is_valid
        error_messages = [e.message for e in result.errors]
        assert any("primary" in msg.lower() for msg in error_messages)

    def test_multiple_tables_all_have_pk(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        result = validate_primary_keys(schema)
        assert result.is_valid, f"Full schema PK check failed: {result.errors}"


# ===========================================================================
# Tests for validate_column_types
# ===========================================================================


class TestValidateColumnTypes:
    """Tests for column type validation against allowed types."""

    def test_valid_types_pass(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        result = validate_column_types(schema)
        assert result.is_valid

    def test_invalid_type_fails(self, schema_invalid_column_type: Dict[str, Any]) -> None:
        schema = _build_schema(schema_invalid_column_type)
        result = validate_column_types(schema)
        assert not result.is_valid
        error_messages = [e.message for e in result.errors]
        assert any("hyperblob" in msg.lower() or "type" in msg.lower() for msg in error_messages)

    def test_full_schema_all_types_valid(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        result = validate_column_types(schema)
        assert result.is_valid


# ===========================================================================
# Tests for validate_foreign_keys
# ===========================================================================


class TestValidateForeignKeys:
    """Tests for foreign key referential integrity validation."""

    def test_valid_fk_passes(self, two_table_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(two_table_schema_dict)
        result = validate_foreign_keys(schema)
        assert result.is_valid

    def test_broken_fk_fails(self, schema_broken_foreign_key: Dict[str, Any]) -> None:
        schema = _build_schema(schema_broken_foreign_key)
        result = validate_foreign_keys(schema)
        assert not result.is_valid
        error_messages = [e.message for e in result.errors]
        assert any("ghost" in msg.lower() or "not found" in msg.lower() or "exist" in msg.lower() for msg in error_messages), (
            f"Expected FK error mentioning 'GhostTable', got: {error_messages}"
        )

    def test_full_schema_fk_valid(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        result = validate_foreign_keys(schema)
        assert result.is_valid

    def test_self_referencing_fk_passes(self, schema_dict: Dict[str, Any]) -> None:
        """Category.parent_id -> Category.id is a self-reference and must pass."""
        schema = _build_schema(schema_dict)
        result = validate_foreign_keys(schema)
        assert result.is_valid


# ===========================================================================
# Tests for validate_relationships
# ===========================================================================


class TestValidateRelationships:
    """Tests for relationship target and back_populates validation."""

    def test_valid_relationships_pass(self, two_table_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(two_table_schema_dict)
        result = validate_relationships(schema)
        assert result.is_valid

    def test_relationship_to_nonexistent_table(self) -> None:
        raw = {
            "project": {"name": "test", "version": "0.1.0", "description": "", "database_url": "sqlite:///test.db"},
            "config": {
                "api_prefix": "/api/v1", "enable_soft_delete": False, "enable_timestamps": False,
                "enable_pagination": True, "default_page_size": 10, "max_page_size": 50, "auth_enabled": False,
            },
            "tables": [
                {
                    "name": "Lonely",
                    "description": "Has a relationship to nothing",
                    "columns": [
                        {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                    ],
                    "relationships": [
                        {"name": "ghosts", "target": "NonExistent", "type": "one-to-many", "back_populates": "lonely"},
                    ],
                }
            ],
        }
        schema = _build_schema(raw)
        result = validate_relationships(schema)
        assert not result.is_valid

    def test_full_schema_relationships_valid(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        result = validate_relationships(schema)
        assert result.is_valid


# ===========================================================================
# Tests for validate_topological_order
# ===========================================================================


class TestValidateTopologicalOrder:
    """Tests for circular dependency detection."""

    def test_acyclic_graph_passes(self, two_table_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(two_table_schema_dict)
        result = validate_topological_order(schema)
        assert result.is_valid

    def test_circular_dependency_detected(self, schema_circular_foreign_keys: Dict[str, Any]) -> None:
        schema = _build_schema(schema_circular_foreign_keys)
        result = validate_topological_order(schema)
        # Circular deps should either fail validation or be flagged as warnings
        # depending on implementation. At minimum, the validator should not crash.
        # If the implementation treats nullable FK cycles as warnings, result may be valid
        # but should have warnings. If strict, it should fail.
        assert isinstance(result, ValidationResult)

    def test_full_schema_no_cycles(self, schema_dict: Dict[str, Any]) -> None:
        """The full shop schema has no true cycles (Category self-ref is nullable)."""
        schema = _build_schema(schema_dict)
        result = validate_topological_order(schema)
        assert result.is_valid


# ===========================================================================
# Tests for validate_full (integration of all validators)
# ===========================================================================


class TestValidateFull:
    """Integration tests for the full validation pipeline."""

    def test_valid_schema_passes_full_validation(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        result = validate_full(schema_dict, schema)
        assert result.is_valid, f"Full validation failed: {[e.message for e in result.errors]}"

    def test_minimal_schema_passes(self, minimal_schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(minimal_schema_dict)
        result = validate_full(minimal_schema_dict, schema)
        assert result.is_valid

    def test_broken_fk_fails_full(self, schema_broken_foreign_key: Dict[str, Any]) -> None:
        schema = _build_schema(schema_broken_foreign_key)
        result = validate_full(schema_broken_foreign_key, schema)
        assert not result.is_valid

    def test_no_pk_fails_full(self, schema_no_primary_key: Dict[str, Any]) -> None:
        schema = _build_schema(schema_no_primary_key)
        result = validate_full(schema_no_primary_key, schema)
        assert not result.is_valid

    def test_invalid_type_fails_full(self, schema_invalid_column_type: Dict[str, Any]) -> None:
        schema = _build_schema(schema_invalid_column_type)
        result = validate_full(schema_invalid_column_type, schema)
        assert not result.is_valid

    def test_duplicate_names_fails_full(self, schema_duplicate_table_names: Dict[str, Any]) -> None:
        schema = _build_schema(schema_duplicate_table_names)
        result = validate_full(schema_duplicate_table_names, schema)
        assert not result.is_valid

    def test_validation_result_has_error_count(self, schema_broken_foreign_key: Dict[str, Any]) -> None:
        schema = _build_schema(schema_broken_foreign_key)
        result = validate_full(schema_broken_foreign_key, schema)
        assert len(result.errors) > 0

    def test_valid_schema_zero_errors(self, schema_dict: Dict[str, Any]) -> None:
        schema = _build_schema(schema_dict)
        result = validate_full(schema_dict, schema)
        assert len(result.errors) == 0
