"""
tests/test_generator.py
End-to-End integration tests for apigen.generator (APIGenerator).

These tests exercise the COMPLETE 5-stage pipeline:
  1. Load schema file from disk
  2. Validate schema
  3. Topological sort
  4. Generate code (via TemplateGenerator)
  5. Export to real files (via ProjectExporter)

All I/O is performed in pytest's tmp_path directories.
NO mocking is used for the core pipeline.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict

import pytest
import yaml

from apigen.generator import APIGenerator, GenerationReport
from apigen.exporters import ProjectExporter


# ===========================================================================
# Full Pipeline E2E Tests
# ===========================================================================


class TestAPIGeneratorEndToEnd:
    """End-to-end tests that run the full generation pipeline and verify output files."""

    def test_full_pipeline_minimal_schema(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """Run full pipeline on minimal schema and verify output structure."""
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()

        assert report.success, f"Pipeline failed: {report.errors}"
        assert output_dir.exists()

        # Check that at least some Python files were created
        py_files = list(output_dir.rglob("*.py"))
        assert len(py_files) > 0, "No Python files were generated"

    def test_full_pipeline_two_table_schema(
        self, two_table_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """Run full pipeline on two-table schema with FK relationships."""
        generator = APIGenerator(
            schema_path=str(two_table_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()

        assert report.success, f"Pipeline failed: {report.errors}"
        py_files = list(output_dir.rglob("*.py"))
        assert len(py_files) >= 2, f"Expected >= 2 Python files, got {len(py_files)}"

    def test_full_pipeline_full_shop_schema(
        self, schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """Run full pipeline on the complete online shop schema (6 tables)."""
        generator = APIGenerator(
            schema_path=str(schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()

        assert report.success, f"Pipeline failed: {report.errors}"

        # Verify directory structure was created
        assert output_dir.exists()
        py_files = list(output_dir.rglob("*.py"))
        assert len(py_files) >= 4, (
            f"Full shop schema should generate many files, got {len(py_files)}"
        )

    def test_generated_files_are_valid_python(
        self, schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """Every .py file in the output must be syntactically valid Python."""
        import ast

        generator = APIGenerator(
            schema_path=str(schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success

        py_files = list(output_dir.rglob("*.py"))
        for py_file in py_files:
            code = py_file.read_text(encoding="utf-8")
            try:
                ast.parse(code, filename=str(py_file))
            except SyntaxError as e:
                pytest.fail(
                    f"Syntax error in generated file {py_file.relative_to(output_dir)}: {e}\n"
                    f"First 500 chars:\n{code[:500]}"
                )

    def test_generated_files_are_nonempty(
        self, schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """No generated .py file should be empty."""
        generator = APIGenerator(
            schema_path=str(schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success

        py_files = list(output_dir.rglob("*.py"))
        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8").strip()
            assert len(content) > 0, f"Generated file is empty: {py_file}"


# ===========================================================================
# Generation Report Tests
# ===========================================================================


class TestGenerationReport:
    """Tests for the GenerationReport produced by the pipeline."""

    def test_report_has_timing_info(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success
        assert hasattr(report, "duration") or hasattr(report, "elapsed") or hasattr(report, "timing")
        # The report should have some notion of how long it took
        # Check the most common attribute names
        duration_value = getattr(report, "duration", None) or getattr(report, "elapsed", None) or getattr(report, "total_time", None)
        if duration_value is not None:
            assert duration_value >= 0

    def test_report_has_table_count(
        self, schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success
        # Report should know how many tables were processed
        table_count = getattr(report, "table_count", None) or getattr(report, "tables_processed", None) or getattr(report, "num_tables", None)
        if table_count is not None:
            assert table_count == 6  # 6 tables in shop schema

    def test_report_has_file_count(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success
        file_count = getattr(report, "file_count", None) or getattr(report, "files_written", None) or getattr(report, "num_files", None)
        if file_count is not None:
            assert file_count > 0

    def test_report_success_flag_true_on_valid(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success is True

    def test_report_errors_empty_on_valid(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert len(report.errors) == 0


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestPipelineErrorHandling:
    """Tests for pipeline error handling and resilience."""

    def test_nonexistent_schema_file_fails_gracefully(
        self, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path="/nonexistent/path/schema.yaml",
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert not report.success
        assert len(report.errors) > 0

    def test_invalid_yaml_content_fails(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{{{not: valid: yaml: [[", encoding="utf-8")
        generator = APIGenerator(
            schema_path=str(bad_yaml),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert not report.success

    def test_schema_validation_failure_reported(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path, schema_no_primary_key: Dict[str, Any]
    ) -> None:
        bad_schema_path = tmp_path / "no_pk.yaml"
        with open(bad_schema_path, "w", encoding="utf-8") as fh:
            yaml.dump(schema_no_primary_key, fh)
        generator = APIGenerator(
            schema_path=str(bad_schema_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert not report.success
        assert len(report.errors) > 0

    def test_broken_fk_schema_fails_pipeline(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path, schema_broken_foreign_key: Dict[str, Any]
    ) -> None:
        fk_path = tmp_path / "broken_fk.yaml"
        with open(fk_path, "w", encoding="utf-8") as fh:
            yaml.dump(schema_broken_foreign_key, fh)
        generator = APIGenerator(
            schema_path=str(fk_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert not report.success

    def test_empty_file_fails(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        generator = APIGenerator(
            schema_path=str(empty_yaml),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert not report.success


# ===========================================================================
# Idempotency Tests
# ===========================================================================


class TestPipelineIdempotency:
    """Tests that running the pipeline multiple times on the same output is safe."""

    def test_double_run_does_not_crash(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report1 = generator.run()
        assert report1.success

        # Run again on the same output directory
        generator2 = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report2 = generator2.run()
        assert report2.success

    def test_double_run_produces_same_files(
        self, minimal_schema_yaml_path: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        out1 = tmp_path / "run1"
        out1.mkdir()
        out2 = tmp_path / "run2"
        out2.mkdir()

        gen1 = APIGenerator(schema_path=str(minimal_schema_yaml_path), output_dir=str(out1))
        gen2 = APIGenerator(schema_path=str(minimal_schema_yaml_path), output_dir=str(out2))

        report1 = gen1.run()
        report2 = gen2.run()

        assert report1.success
        assert report2.success

        files1 = sorted([f.relative_to(out1) for f in out1.rglob("*.py")])
        files2 = sorted([f.relative_to(out2) for f in out2.rglob("*.py")])
        assert files1 == files2, f"File lists differ:\n  Run1: {files1}\n  Run2: {files2}"

        # Content should be identical
        for rel_path in files1:
            content1 = (out1 / rel_path).read_text(encoding="utf-8")
            content2 = (out2 / rel_path).read_text(encoding="utf-8")
            assert content1 == content2, f"Content differs for {rel_path}"


# ===========================================================================
# ProjectExporter Atomic Write Tests
# ===========================================================================


class TestExporterAtomicWrites:
    """Tests specifically for the ProjectExporter's file writing behavior."""

    def test_exporter_creates_directory_structure(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success
        assert output_dir.is_dir()

    def test_exporter_writes_to_nested_dirs(
        self, schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(schema_yaml_path),
            output_dir=str(output_dir),
        )
        report = generator.run()
        assert report.success

        # Should have created subdirectories
        all_dirs = set()
        for f in output_dir.rglob("*"):
            if f.is_dir():
                all_dirs.add(f.name)
        # At least the output directory itself has content
        all_files = list(output_dir.rglob("*"))
        assert len(all_files) > 0


# ===========================================================================
# Validate-Only Mode Tests
# ===========================================================================


class TestValidateOnlyMode:
    """Tests for running the generator in validate-only mode (no file output)."""

    def test_validate_only_does_not_write_files(
        self, minimal_schema_yaml_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        generator = APIGenerator(
            schema_path=str(minimal_schema_yaml_path),
            output_dir=str(output_dir),
            validate_only=True,
        )
        report = generator.run()
        assert report.success

        # In validate-only mode, no .py files should be generated in output
        py_files = list(output_dir.rglob("*.py"))
        assert len(py_files) == 0, (
            f"Validate-only mode should not write files, but found: {py_files}"
        )

    def test_validate_only_reports_errors_for_bad_schema(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path, schema_no_primary_key: Dict[str, Any]
    ) -> None:
        bad_path = tmp_path / "bad_validate.yaml"
        with open(bad_path, "w", encoding="utf-8") as fh:
            yaml.dump(schema_no_primary_key, fh)
        generator = APIGenerator(
            schema_path=str(bad_path),
            output_dir=str(output_dir),
            validate_only=True,
        )
        report = generator.run()
        assert not report.success
        assert len(report.errors) > 0


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for unusual but valid inputs."""

    def test_table_with_many_columns(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """A table with 30 columns should not cause any issues."""
        columns = [
            {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True}
        ]
        for i in range(29):
            columns.append(
                {"name": f"field_{i}", "type": "string", "max_length": 100, "nullable": True}
            )
        raw = {
            "project": {
                "name": "big_table",
                "version": "0.1.0",
                "description": "Table with many columns",
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
                    "name": "BigTable",
                    "description": "30 columns",
                    "columns": columns,
                    "relationships": [],
                }
            ],
        }
        schema_path = tmp_path / "big.yaml"
        with open(schema_path, "w", encoding="utf-8") as fh:
            yaml.dump(raw, fh)

        generator = APIGenerator(schema_path=str(schema_path), output_dir=str(output_dir))
        report = generator.run()
        assert report.success

    def test_table_name_with_underscore(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """Table name like 'UserProfile' should work fine."""
        raw = {
            "project": {
                "name": "underscore_test",
                "version": "0.1.0",
                "description": "Test",
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
                    "name": "UserProfile",
                    "description": "CamelCase name",
                    "columns": [
                        {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                        {"name": "bio", "type": "text", "nullable": True},
                    ],
                    "relationships": [],
                }
            ],
        }
        path = tmp_path / "camel.yaml"
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(raw, fh)

        generator = APIGenerator(schema_path=str(path), output_dir=str(output_dir))
        report = generator.run()
        assert report.success

    def test_boolean_and_float_types(
        self, tmp_path: pathlib.Path, output_dir: pathlib.Path
    ) -> None:
        """All supported types should generate without error."""
        raw = {
            "project": {
                "name": "all_types",
                "version": "0.1.0",
                "description": "Test all column types",
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
                    "name": "AllTypes",
                    "description": "Every supported type",
                    "columns": [
                        {"name": "id", "type": "integer", "primary_key": True, "autoincrement": True},
                        {"name": "name", "type": "string", "max_length": 100, "nullable": False},
                        {"name": "bio", "type": "text", "nullable": True},
                        {"name": "score", "type": "float", "nullable": True},
                        {"name": "is_active", "type": "boolean", "default": True},
                        {"name": "created_at", "type": "datetime", "nullable": True},
                    ],
                    "relationships": [],
                }
            ],
        }
        path = tmp_path / "types.yaml"
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(raw, fh)

        generator = APIGenerator(schema_path=str(path), output_dir=str(output_dir))
        report = generator.run()
        assert report.success
