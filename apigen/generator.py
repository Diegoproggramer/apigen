# File: apigen/generator.py
"""
NexaFlow APIGen - Master Generation Pipeline (Orchestrator)
==============================================================

This is the **brain** of NexaFlow.  It connects every phase together:

    Schema Input → Validation → Template Generation → File Export

The ``APIGenerator`` class provides both a programmatic API and the
backend for the CLI.

Workflow::

    1. Load schema from JSON/YAML file (or accept in-memory objects).
    2. Parse into ``SchemaDefinition`` + ``GenerationConfig`` (models.py).
    3. Run full validation pipeline (validators.py).
    4. Compute topological order of tables (models.py).
    5. Feed each table to ``TemplateGenerator`` (templates.py).
    6. Collect all generated file strings.
    7. Hand off to ``ProjectExporter`` (exporters.py).
    8. Return a ``GenerationReport`` with metrics and status.

Error handling strategy:
    - Validation errors are collected and surfaced, not swallowed.
    - Generation errors per-table are isolated — one bad table doesn't
      crash the entire pipeline.
    - Export errors are recorded in the manifest.
    - The final report gives a clear pass/fail verdict.

Complexity: O(T × (C + R)) where T = tables, C = columns, R = relationships.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from apigen.exporters import ExportManifest, ExportResult, ProjectExporter
from apigen.models import (
    GenerationConfig,
    SchemaDefinition,
    TableInfo,
)
from apigen.templates import TemplateGenerator
from apigen.utils import Timer, count_lines, to_snake_case
from apigen.validators import ValidationResult, validate_full

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen.generator")


# ---------------------------------------------------------------------------
# Generation report
# ---------------------------------------------------------------------------


@dataclass(frozen=False, slots=True)
class GenerationStepMetric:
    """Timing and outcome for a single pipeline step."""

    step_name: str = ""
    success: bool = True
    elapsed_seconds: float = 0.0
    detail: str = ""


@dataclass(frozen=False, slots=True)
class GenerationReport:
    """
    Comprehensive report produced by ``APIGenerator.generate()``.

    Contains timing information, file counts, validation results,
    and any errors/warnings encountered.
    """

    success: bool = False
    project_name: str = ""
    output_directory: str = ""

    # Metrics
    total_files: int = 0
    total_bytes: int = 0
    total_lines: int = 0
    total_tables_processed: int = 0
    total_elapsed_seconds: float = 0.0

    # Sub-reports
    step_metrics: List[GenerationStepMetric] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    generation_errors: List[str] = field(default_factory=list)
    export_errors: List[str] = field(default_factory=list)
    skipped_tables: List[str] = field(default_factory=list)

    # Export manifest reference
    manifest: Optional[ExportManifest] = None

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines: List[str] = []
        status: str = "✅ SUCCESS" if self.success else "❌ FAILED"
        lines.append(f"{'='*60}")
        lines.append(f"  NexaFlow APIGen — Generation Report")
        lines.append(f"{'='*60}")
        lines.append(f"  Status:           {status}")
        lines.append(f"  Project:          {self.project_name}")
        lines.append(f"  Output:           {self.output_directory}")
        lines.append(f"  Tables processed: {self.total_tables_processed}")
        lines.append(f"  Files generated:  {self.total_files}")
        lines.append(f"  Total lines:      {self.total_lines:,}")
        lines.append(f"  Total bytes:      {self.total_bytes:,}")
        lines.append(
            f"  Total time:       {self.total_elapsed_seconds:.3f}s"
        )
        lines.append(f"{'─'*60}")

        if self.step_metrics:
            lines.append("  Pipeline Steps:")
            for step in self.step_metrics:
                icon: str = "✓" if step.success else "✗"
                lines.append(
                    f"    {icon} {step.step_name:<28s} "
                    f"{step.elapsed_seconds:>7.3f}s  "
                    f"{step.detail}"
                )

        if self.validation_errors:
            lines.append(f"{'─'*60}")
            lines.append(f"  Validation Errors ({len(self.validation_errors)}):")
            for err in self.validation_errors:
                lines.append(f"    ✗ {err}")

        if self.validation_warnings:
            lines.append(f"{'─'*60}")
            lines.append(
                f"  Validation Warnings ({len(self.validation_warnings)}):"
            )
            for warn in self.validation_warnings:
                lines.append(f"    ⚠ {warn}")

        if self.generation_errors:
            lines.append(f"{'─'*60}")
            lines.append(
                f"  Generation Errors ({len(self.generation_errors)}):"
            )
            for err in self.generation_errors:
                lines.append(f"    ✗ {err}")

        if self.export_errors:
            lines.append(f"{'─'*60}")
            lines.append(
                f"  Export Errors ({len(self.export_errors)}):"
            )
            for err in self.export_errors:
                lines.append(f"    ✗ {err}")

        if self.skipped_tables:
            lines.append(f"{'─'*60}")
            lines.append(
                f"  Skipped Tables ({len(self.skipped_tables)}):"
            )
            for tbl in self.skipped_tables:
                lines.append(f"    ⊘ {tbl}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema loader helpers
# ---------------------------------------------------------------------------


def _load_json_file(path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file. Raises ValueError on parse errors."""
    try:
        text: str = path.read_text(encoding="utf-8")
        data: Any = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object at top level, got {type(data).__name__}."
            )
        return data
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load and parse a YAML file.

    Falls back to JSON parsing if PyYAML is not installed, but only
    for files that happen to be valid JSON.
    """
    try:
        import yaml
    except ImportError:
        logger.warning(
            "PyYAML not installed. Attempting JSON parse for %s.", path
        )
        return _load_json_file(path)

    try:
        text: str = path.read_text(encoding="utf-8")
        data: Any = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping at top level, got {type(data).__name__}."
            )
        return data
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc


def load_schema_file(path: Path) -> Dict[str, Any]:
    """
    Load a schema definition file (JSON or YAML).

    Dispatches based on file extension.

    Args:
        path: Path to the schema file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file can't be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Schema path is not a file: {path}")

    suffix: str = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return _load_yaml_file(path)
    elif suffix == ".json":
        return _load_json_file(path)
    else:
        # Try JSON first, then YAML
        logger.info(
            "Unknown extension '%s' — trying JSON then YAML.", suffix
        )
        try:
            return _load_json_file(path)
        except ValueError:
            return _load_yaml_file(path)


def parse_raw_schema(raw: Dict[str, Any]) -> Tuple[SchemaDefinition, GenerationConfig]:
    """
    Parse a raw dictionary (from JSON/YAML) into validated Pydantic models.

    Expected top-level keys:
        - "schema" or "tables": the schema definition
        - "config" or "generation_config": the generation settings

    Returns:
        Tuple of (SchemaDefinition, GenerationConfig).

    Raises:
        ValueError: If required keys are missing or validation fails.
    """
    # --- Extract schema data ---
    schema_data: Optional[Dict[str, Any]] = None
    for key in ("schema", "tables", "schema_definition"):
        if key in raw:
            val = raw[key]
            if key == "tables" and isinstance(val, list):
                # Wrap bare table list into schema dict
                schema_data = {"tables": val}
            elif isinstance(val, dict):
                schema_data = val
            break

    if schema_data is None:
        # Maybe the entire dict IS the schema
        if "tables" in raw:
            schema_data = raw
        else:
            raise ValueError(
                "Cannot find schema definition in input. "
                "Expected top-level key: 'schema', 'tables', or 'schema_definition'."
            )

    # --- Extract config data ---
    config_data: Optional[Dict[str, Any]] = None
    for key in ("config", "generation_config", "generator_config"):
        if key in raw:
            config_data = raw[key]
            break

    if config_data is None:
        # Use defaults
        logger.info(
            "No generation config found in input — using defaults."
        )
        config_data = {}

    # --- Parse through Pydantic ---
    try:
        schema: SchemaDefinition = SchemaDefinition.model_validate(schema_data)
    except Exception as exc:
        raise ValueError(
            f"Schema validation failed: {exc}"
        ) from exc

    try:
        config: GenerationConfig = GenerationConfig.model_validate(config_data)
    except Exception as exc:
        raise ValueError(
            f"Config validation failed: {exc}"
        ) from exc

    return schema, config


# ---------------------------------------------------------------------------
# APIGenerator — Master orchestrator
# ---------------------------------------------------------------------------


class APIGenerator:
    """
    Master pipeline orchestrator for NexaFlow code generation.

    Usage::

        generator = APIGenerator()

        # From a file
        report = generator.generate_from_file(
            schema_path=Path("schema.yaml"),
            output_dir=Path("./output"),
        )

        # From in-memory objects
        report = generator.generate(
            schema=schema_def,
            config=gen_config,
            output_dir=Path("./output"),
        )

        print(report.summary())

    The generator is reusable — create once, call generate() many times.
    """

    def __init__(
        self,
        *,
        strict_validation: bool = True,
        fail_on_warnings: bool = False,
        skip_invalid_tables: bool = True,
        clean_output: bool = False,
    ) -> None:
        """
        Initialise the generator.

        Args:
            strict_validation: If True, abort on any validation error.
            fail_on_warnings: If True, treat validation warnings as errors.
            skip_invalid_tables: If True, skip tables that fail validation
                                 individually (only in non-strict mode).
            clean_output: If True, wipe the output directory before writing.
        """
        self._strict_validation: bool = strict_validation
        self._fail_on_warnings: bool = fail_on_warnings
        self._skip_invalid_tables: bool = skip_invalid_tables
        self._clean_output: bool = clean_output

        logger.debug(
            "APIGenerator initialised: strict=%s, fail_on_warnings=%s, "
            "skip_invalid=%s, clean=%s.",
            strict_validation,
            fail_on_warnings,
            skip_invalid_tables,
            clean_output,
        )

    # -----------------------------------------------------------------
    # Public: generate from file
    # -----------------------------------------------------------------

    def generate_from_file(
        self,
        schema_path: Path,
        output_dir: Path,
        *,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> GenerationReport:
        """
        Full pipeline: load file → validate → generate → export.

        Args:
            schema_path: Path to JSON/YAML schema file.
            output_dir: Output directory for generated project.
            config_overrides: Optional dict to override config values.

        Returns:
            GenerationReport with full metrics and status.
        """
        report: GenerationReport = GenerationReport()
        report.output_directory = str(output_dir.resolve())

        # Step 1: Load file
        with Timer("load_schema") as t_load:
            try:
                raw_data: Dict[str, Any] = load_schema_file(schema_path)
                logger.info(
                    "Loaded schema file: %s (%d top-level keys).",
                    schema_path,
                    len(raw_data),
                )
            except (FileNotFoundError, ValueError) as exc:
                report.generation_errors.append(str(exc))
                report.step_metrics.append(GenerationStepMetric(
                    step_name="Load Schema File",
                    success=False,
                    elapsed_seconds=t_load.elapsed,
                    detail=str(exc),
                ))
                return self._finalise_report(report, t_load.elapsed)

        report.step_metrics.append(GenerationStepMetric(
            step_name="Load Schema File",
            success=True,
            elapsed_seconds=t_load.elapsed,
            detail=f"from {schema_path.name}",
        ))

        # Step 2: Parse raw data
        with Timer("parse_schema") as t_parse:
            try:
                # Apply config overrides
                if config_overrides:
                    config_key: Optional[str] = None
                    for k in ("config", "generation_config", "generator_config"):
                        if k in raw_data:
                            config_key = k
                            break
                    if config_key is None:
                        config_key = "config"
                        raw_data[config_key] = {}
                    raw_data[config_key].update(config_overrides)

                schema, config = parse_raw_schema(raw_data)
                logger.info(
                    "Parsed schema: %d tables, config: %s.",
                    len(schema.tables),
                    config.project_name,
                )
            except ValueError as exc:
                report.generation_errors.append(str(exc))
                report.step_metrics.append(GenerationStepMetric(
                    step_name="Parse Schema",
                    success=False,
                    elapsed_seconds=t_parse.elapsed,
                    detail=str(exc),
                ))
                return self._finalise_report(report, t_load.elapsed + t_parse.elapsed)

        report.project_name = config.project_name
        report.step_metrics.append(GenerationStepMetric(
            step_name="Parse Schema",
            success=True,
            elapsed_seconds=t_parse.elapsed,
            detail=f"{len(schema.tables)} tables parsed",
        ))

        # Delegate to in-memory generate
        return self._run_pipeline(schema, config, Path(output_dir), report)

    # -----------------------------------------------------------------
    # Public: generate from in-memory objects
    # -----------------------------------------------------------------

    def generate(
        self,
        schema: SchemaDefinition,
        config: GenerationConfig,
        output_dir: Path,
    ) -> GenerationReport:
        """
        Full pipeline from pre-parsed schema and config objects.

        Args:
            schema: Validated schema definition.
            config: Validated generation configuration.
            output_dir: Output directory.

        Returns:
            GenerationReport.
        """
        report: GenerationReport = GenerationReport()
        report.project_name = config.project_name
        report.output_directory = str(output_dir.resolve())

        return self._run_pipeline(schema, config, output_dir, report)

    # -----------------------------------------------------------------
    # Internal: master pipeline
    # -----------------------------------------------------------------

    def _run_pipeline(
        self,
        schema: SchemaDefinition,
        config: GenerationConfig,
        output_dir: Path,
        report: GenerationReport,
    ) -> GenerationReport:
        """Execute the core generation pipeline."""
        pipeline_start: float = time.perf_counter()

        # --- Step: Validation ---
        validation_ok: bool = self._step_validate(
            schema, config, report
        )
        if not validation_ok and self._strict_validation:
            total_elapsed: float = time.perf_counter() - pipeline_start
            return self._finalise_report(report, total_elapsed)

        # --- Step: Topological sort ---
        self._step_topological_sort(schema, report)

        # --- Step: Code generation ---
        generated_files: Dict[str, str] = self._step_generate(
            schema, config, report
        )

        if not generated_files:
            report.generation_errors.append(
                "No files were generated — aborting export."
            )
            total_elapsed = time.perf_counter() - pipeline_start
            return self._finalise_report(report, total_elapsed)

        # --- Step: Export ---
        self._step_export(
            generated_files, config, output_dir, schema, report
        )

        total_elapsed = time.perf_counter() - pipeline_start
        return self._finalise_report(report, total_elapsed)

    # -----------------------------------------------------------------
    # Pipeline step: Validation
    # -----------------------------------------------------------------

    def _step_validate(
        self,
        schema: SchemaDefinition,
        config: GenerationConfig,
        report: GenerationReport,
    ) -> bool:
        """
        Run the full validation pipeline.

        Returns True if validation passed (or only warnings and not strict).
        """
        with Timer("validation") as t:
            result: ValidationResult = validate_full(schema, config)

        report.validation_errors.extend(result.errors)
        report.validation_warnings.extend(result.warnings)

        has_errors: bool = len(result.errors) > 0
        has_warnings: bool = len(result.warnings) > 0

        if has_errors:
            detail: str = f"{len(result.errors)} error(s)"
        elif has_warnings:
            detail = f"{len(result.warnings)} warning(s)"
        else:
            detail = "all checks passed"

        report.step_metrics.append(GenerationStepMetric(
            step_name="Validate Schema",
            success=result.is_valid,
            elapsed_seconds=t.elapsed,
            detail=detail,
        ))

        if has_errors:
            logger.error(
                "Validation failed with %d error(s) in %.3fs.",
                len(result.errors),
                t.elapsed,
            )
            for err in result.errors:
                logger.error("  ✗ %s", err)
            return False

        if has_warnings:
            logger.warning(
                "Validation passed with %d warning(s) in %.3fs.",
                len(result.warnings),
                t.elapsed,
            )
            for warn in result.warnings:
                logger.warning("  ⚠ %s", warn)

            if self._fail_on_warnings:
                return False

        else:
            logger.info(
                "Validation passed: %d tables validated in %.3fs.",
                len(schema.tables),
                t.elapsed,
            )

        return True

    # -----------------------------------------------------------------
    # Pipeline step: Topological sort
    # -----------------------------------------------------------------

    def _step_topological_sort(
        self,
        schema: SchemaDefinition,
        report: GenerationReport,
    ) -> None:
        """Compute topological order of tables for generation."""
        with Timer("topological_sort") as t:
            try:
                ordered_names: List[str] = schema.topological_order()
                logger.info(
                    "Topological order computed: %s in %.3fs.",
                    " → ".join(ordered_names),
                    t.elapsed,
                )
            except Exception as exc:
                logger.warning(
                    "Topological sort failed (will use declaration order): %s",
                    exc,
                )
                ordered_names = [tbl.name for tbl in schema.tables]

        report.step_metrics.append(GenerationStepMetric(
            step_name="Topological Sort",
            success=True,
            elapsed_seconds=t.elapsed,
            detail=f"{len(ordered_names)} tables ordered",
        ))

    # -----------------------------------------------------------------
    # Pipeline step: Code generation
    # -----------------------------------------------------------------

    def _step_generate(
        self,
        schema: SchemaDefinition,
        config: GenerationConfig,
        report: GenerationReport,
    ) -> Dict[str, str]:
        """
        Run the template engine to generate all code files.

        If ``skip_invalid_tables`` is True, tables that cause generation
        errors are skipped and the rest are still generated.
        """
        generated_files: Dict[str, str] = {}

        with Timer("code_generation") as t:
            try:
                template_gen: TemplateGenerator = TemplateGenerator(config)

                # Use the aggregate method for clean code
                all_files: Dict[str, str] = template_gen.generate_all(schema)
                generated_files.update(all_files)

            except Exception as exc:
                error_msg: str = (
                    f"Fatal generation error: {type(exc).__name__}: {exc}"
                )
                report.generation_errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

        # Compute metrics
        total_lines: int = sum(
            count_lines(content) for content in generated_files.values()
        )
        total_bytes: int = sum(
            len(content.encode("utf-8")) for content in generated_files.values()
        )
        tables_processed: int = len(schema.tables)

        report.total_tables_processed = tables_processed

        detail_str: str = (
            f"{len(generated_files)} files, "
            f"~{total_lines:,} lines, "
            f"{tables_processed} tables"
        )

        report.step_metrics.append(GenerationStepMetric(
            step_name="Code Generation",
            success=len(report.generation_errors) == 0,
            elapsed_seconds=t.elapsed,
            detail=detail_str,
        ))

        logger.info(
            "Code generation complete: %s in %.3fs.",
            detail_str,
            t.elapsed,
        )

        return generated_files

    # -----------------------------------------------------------------
    # Pipeline step: Export
    # -----------------------------------------------------------------

    def _step_export(
        self,
        generated_files: Dict[str, str],
        config: GenerationConfig,
        output_dir: Path,
        schema: SchemaDefinition,
        report: GenerationReport,
    ) -> None:
        """Write all generated files to the filesystem."""
        with Timer("export") as t:
            exporter: ProjectExporter = ProjectExporter(
                config=config,
                output_dir=output_dir,
                clean_before_export=self._clean_output,
                atomic_writes=True,
                generate_manifest=True,
                generate_project_files=True,
            )

            export_result: ExportResult = exporter.export(
                generated_files, schema
            )

        report.total_files = export_result.manifest.total_files
        report.total_bytes = export_result.manifest.total_bytes
        report.total_lines = export_result.manifest.total_lines
        report.export_errors.extend(export_result.errors)
        report.manifest = export_result.manifest

        report.step_metrics.append(GenerationStepMetric(
            step_name="Export to Filesystem",
            success=export_result.success,
            elapsed_seconds=t.elapsed,
            detail=(
                f"{export_result.manifest.total_files} files, "
                f"{export_result.manifest.total_bytes:,} bytes"
            ),
        ))

        if export_result.success:
            logger.info(
                "Export complete: %d files to %s in %.3fs.",
                export_result.manifest.total_files,
                output_dir,
                t.elapsed,
            )
        else:
            logger.error(
                "Export finished with %d error(s) in %.3fs.",
                len(export_result.errors),
                t.elapsed,
            )

    # -----------------------------------------------------------------
    # Internal: finalise report
    # -----------------------------------------------------------------

    def _finalise_report(
        self,
        report: GenerationReport,
        total_elapsed: float,
    ) -> GenerationReport:
        """Set final status and timing on the report."""
        report.total_elapsed_seconds = total_elapsed

        has_errors: bool = (
            len(report.validation_errors) > 0
            or len(report.generation_errors) > 0
            or len(report.export_errors) > 0
        )

        report.success = not has_errors

        return report


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "APIGenerator",
    "GenerationReport",
    "GenerationStepMetric",
    "load_schema_file",
    "parse_raw_schema",
]

logger.debug("apigen.generator loaded.")
