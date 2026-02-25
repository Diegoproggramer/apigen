# File: apigen/cli.py
"""
NexaFlow APIGen - Command-Line Interface
==========================================

Professional CLI built exclusively with the standard-library ``argparse``
module — zero external dependencies.

Usage examples::

    # Basic generation
    python -m apigen --schema schema.yaml --output ./my_api

    # Verbose output with clean directory
    python -m apigen -s schema.json -o ./output --verbose --clean

    # Override project name and use async mode
    python -m apigen -s schema.yaml -o ./out \\
        --project-name "MyApp" --async --dialect postgresql

    # Validate only (no file output)
    python -m apigen -s schema.yaml --validate-only

    # Show version
    python -m apigen --version

Exit codes:
    0 — success
    1 — validation error
    2 — generation error
    3 — export error
    4 — input/argument error
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Sequence

# ---------------------------------------------------------------------------
# Logger (configured in _setup_logging)
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen")


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_SUCCESS: int = 0
EXIT_VALIDATION_ERROR: int = 1
EXIT_GENERATION_ERROR: int = 2
EXIT_EXPORT_ERROR: int = 3
EXIT_INPUT_ERROR: int = 4


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbosity: int) -> None:
    """
    Configure the root apigen logger based on verbosity level.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, 2+ = DEBUG.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Create a nice handler with colour-like formatting
    handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt: str = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
    handler.setFormatter(formatter)

    # Configure the root apigen logger
    root_logger: logging.Logger = logging.getLogger("apigen")
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplication
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    from apigen import __version__

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="apigen",
        description=(
            "NexaFlow APIGen — Automated REST API Code Generator.\n\n"
            "Transforms database schema definitions (JSON/YAML) into "
            "complete, production-ready FastAPI + SQLAlchemy 2.0 applications."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -s schema.yaml -o ./my_api\n"
            "  %(prog)s -s schema.json -o ./out --verbose --clean\n"
            "  %(prog)s -s schema.yaml --validate-only\n"
            "  %(prog)s -s schema.yaml -o ./out --async --dialect postgresql\n"
        ),
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"NexaFlow APIGen v{__version__}",
    )

    # --- Required arguments ---
    parser.add_argument(
        "-s", "--schema",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the schema definition file (JSON or YAML).",
    )

    # --- Output ---
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Output directory for the generated project. "
            "Required unless --validate-only is set."
        ),
    )

    # --- Modes ---
    mode_group = parser.add_argument_group("operation modes")
    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Only validate the schema without generating code.",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Run the full pipeline but don't write files to disk. "
            "Useful for checking generation without side effects."
        ),
    )

    # --- Config overrides ---
    config_group = parser.add_argument_group("configuration overrides")
    config_group.add_argument(
        "--project-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override the project name.",
    )
    config_group.add_argument(
        "--project-version",
        type=str,
        default=None,
        metavar="VER",
        help="Override the project version (e.g. '2.0.0').",
    )
    config_group.add_argument(
        "--dialect",
        type=str,
        default=None,
        choices=["postgresql", "mysql", "sqlite", "oracle", "mssql"],
        help="Override the database dialect.",
    )
    config_group.add_argument(
        "--database-url",
        type=str,
        default=None,
        metavar="URL",
        help="Override the database URL.",
    )
    config_group.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        default=None,
        help="Force async mode (async engine + sessions).",
    )
    config_group.add_argument(
        "--sync",
        dest="use_sync",
        action="store_true",
        default=False,
        help="Force sync mode.",
    )
    config_group.add_argument(
        "--pagination",
        type=str,
        default=None,
        choices=["offset", "page_number", "cursor"],
        help="Override pagination style.",
    )
    config_group.add_argument(
        "--api-prefix",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Override API URL prefix (e.g. '/api/v1').",
    )

    # --- Behaviour flags ---
    behaviour_group = parser.add_argument_group("behaviour flags")
    behaviour_group.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean output directory before generation.",
    )
    behaviour_group.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Abort on any validation error (default).",
    )
    behaviour_group.add_argument(
        "--no-strict",
        action="store_true",
        default=False,
        help="Continue even if validation has errors (skip bad tables).",
    )
    behaviour_group.add_argument(
        "--fail-on-warnings",
        action="store_true",
        default=False,
        help="Treat validation warnings as errors.",
    )
    behaviour_group.add_argument(
        "--no-project-files",
        action="store_true",
        default=False,
        help="Skip generating project support files (pyproject.toml, etc.).",
    )

    # --- Verbosity ---
    verbosity_group = parser.add_argument_group("verbosity")
    verbosity_group.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )
    verbosity_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except errors.",
    )

    return parser


# ---------------------------------------------------------------------------
# Config override builder
# ---------------------------------------------------------------------------


def _build_config_overrides(args: argparse.Namespace) -> Dict[str, object]:
    """Build a config override dictionary from CLI arguments."""
    overrides: Dict[str, object] = {}

    if args.project_name is not None:
        overrides["project_name"] = args.project_name

    if args.project_version is not None:
        overrides["project_version"] = args.project_version

    if args.dialect is not None:
        overrides["database_dialect"] = args.dialect

    if args.database_url is not None:
        overrides["database_url"] = args.database_url

    if args.use_async is True:
        overrides["use_async"] = True
    elif args.use_sync is True:
        overrides["use_async"] = False

    if args.pagination is not None:
        overrides["pagination_style"] = args.pagination

    if args.api_prefix is not None:
        overrides["api_prefix"] = args.api_prefix

    return overrides


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------


def _run_validate_only(schema_path: Path) -> int:
    """
    Run validation only (no code generation).

    Returns the appropriate exit code.
    """
    from apigen.generator import load_schema_file, parse_raw_schema
    from apigen.validators import validate_full

    logger.info("Running validation-only mode for: %s", schema_path)

    # Load
    try:
        raw_data = load_schema_file(schema_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load schema: %s", exc)
        return EXIT_INPUT_ERROR

    # Parse
    try:
        schema, config = parse_raw_schema(raw_data)
    except ValueError as exc:
        logger.error("Failed to parse schema: %s", exc)
        return EXIT_INPUT_ERROR

    # Validate
    from apigen.utils import Timer

    with Timer("validation") as t:
        result = validate_full(schema, config)

    # Report
    print(f"\n{'='*50}")
    print(f"  Schema Validation Report")
    print(f"{'='*50}")
    print(f"  File:     {schema_path.name}")
    print(f"  Tables:   {len(schema.tables)}")
    print(f"  Time:     {t.elapsed:.3f}s")
    print(f"  Valid:    {'Yes' if result.is_valid else 'No'}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    ✗ {err}")

    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for warn in result.warnings:
            print(f"    ⚠ {warn}")

    if result.is_valid and not result.warnings:
        print(f"\n  ✅ All validations passed!")

    print(f"{'='*50}\n")

    return EXIT_SUCCESS if result.is_valid else EXIT_VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Full generation mode
# ---------------------------------------------------------------------------


def _run_generation(
    schema_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """
    Run the full generation pipeline.

    Returns the appropriate exit code.
    """
    from apigen.generator import APIGenerator, GenerationReport

    config_overrides = _build_config_overrides(args)

    strict: bool = not args.no_strict
    generator: APIGenerator = APIGenerator(
        strict_validation=strict,
        fail_on_warnings=args.fail_on_warnings,
        skip_invalid_tables=not strict,
        clean_output=args.clean,
    )

    if args.dry_run:
        logger.info("Dry-run mode: files will not be written to disk.")

    report: GenerationReport = generator.generate_from_file(
        schema_path=schema_path,
        output_dir=output_dir,
        config_overrides=config_overrides if config_overrides else None,
    )

    # Print the summary report
    print(report.summary())

    # Determine exit code
    if not report.success:
        if report.validation_errors:
            return EXIT_VALIDATION_ERROR
        elif report.generation_errors:
            return EXIT_GENERATION_ERROR
        elif report.export_errors:
            return EXIT_EXPORT_ERROR
        return EXIT_GENERATION_ERROR

    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Print the NexaFlow banner."""
    banner: str = r"""
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║    ███╗   ██╗███████╗██╗  ██╗ █████╗             ║
    ║    ████╗  ██║██╔════╝╚██╗██╔╝██╔══██╗            ║
    ║    ██╔██╗ ██║█████╗   ╚███╔╝ ███████║            ║
    ║    ██║╚██╗██║██╔══╝   ██╔██╗ ██╔══██║            ║
    ║    ██║ ╚████║███████╗██╔╝ ██╗██║  ██║            ║
    ║    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝            ║
    ║                                                   ║
    ║    APIGen — Code Generation Engine                ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner, file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def cli_main(argv: Optional[Sequence[str]] = None) -> NoReturn:
    """
    Main CLI entry point.

    Can be called from ``__main__.py`` or directly for testing.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).
    """
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    # --- Verbosity ---
    if args.quiet:
        verbosity: int = -1
        logging.disable(logging.CRITICAL)
    else:
        verbosity = args.verbose

    _setup_logging(verbosity)

    # --- Banner (only if verbose enough) ---
    if verbosity >= 1:
        _print_banner()

    # --- Schema path ---
    schema_path: Path = Path(args.schema).resolve()

    if not schema_path.exists():
        logger.error("Schema file not found: %s", schema_path)
        sys.exit(EXIT_INPUT_ERROR)

    if not schema_path.is_file():
        logger.error("Schema path is not a file: %s", schema_path)
        sys.exit(EXIT_INPUT_ERROR)

    # --- Validate-only mode ---
    if args.validate_only:
        exit_code: int = _run_validate_only(schema_path)
        sys.exit(exit_code)

    # --- Output directory validation ---
    if args.output is None:
        logger.error(
            "Output directory is required for generation. "
            "Use -o/--output or --validate-only."
        )
        parser.print_usage(sys.stderr)
        sys.exit(EXIT_INPUT_ERROR)

    output_dir: Path = Path(args.output).resolve()

    # Log key parameters
    logger.info("Schema:  %s", schema_path)
    logger.info("Output:  %s", output_dir)
    logger.info("Clean:   %s", args.clean)
    logger.info("Strict:  %s", not args.no_strict)

    # --- Run generation ---
    exit_code = _run_generation(schema_path, output_dir, args)

    if exit_code == EXIT_SUCCESS:
        logger.info("Generation completed successfully.")
    else:
        logger.error("Generation failed with exit code %d.", exit_code)

    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "cli_main",
    "EXIT_SUCCESS",
    "EXIT_VALIDATION_ERROR",
    "EXIT_GENERATION_ERROR",
    "EXIT_EXPORT_ERROR",
    "EXIT_INPUT_ERROR",
]

logger.debug("apigen.cli loaded.")
