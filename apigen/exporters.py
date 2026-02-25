# File: apigen/exporters.py
"""
NexaFlow APIGen - Project Exporter (File-System Manager)
==========================================================

Responsible for:
    1. Creating the output directory structure safely.
    2. Writing generated code files atomically (write-to-temp then rename).
    3. Generating supporting project files (pyproject.toml, .env, etc.).
    4. Producing an export manifest with checksums for reproducibility.
    5. Idempotent operation — re-running on the same path is always safe.

All file operations are wrapped with proper error handling.  If a write
fails mid-batch, previously written files remain intact (no cleanup of
partial batches — each individual file is atomic).

Complexity: O(F) where F = total number of output files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from apigen.models import GenerationConfig, SchemaDefinition
from apigen.utils import (
    Timer,
    count_lines,
    ensure_directory,
    sha256_hex,
    to_snake_case,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen.exporters")


# ---------------------------------------------------------------------------
# Data classes for export results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileRecord:
    """Immutable record of a single exported file."""

    relative_path: str
    absolute_path: str
    size_bytes: int
    line_count: int
    sha256: str


@dataclass(frozen=False, slots=True)
class ExportManifest:
    """
    Complete manifest of all exported files.

    Serialisable to JSON for build reproducibility verification.
    """

    project_name: str = ""
    project_version: str = ""
    generator_version: str = ""
    export_timestamp: str = ""
    output_directory: str = ""
    total_files: int = 0
    total_bytes: int = 0
    total_lines: int = 0
    files: List[FileRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a JSON-serialisable dictionary."""
        return {
            "project_name": self.project_name,
            "project_version": self.project_version,
            "generator_version": self.generator_version,
            "export_timestamp": self.export_timestamp,
            "output_directory": self.output_directory,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "total_lines": self.total_lines,
            "files": [
                {
                    "relative_path": f.relative_path,
                    "absolute_path": f.absolute_path,
                    "size_bytes": f.size_bytes,
                    "line_count": f.line_count,
                    "sha256": f.sha256,
                }
                for f in self.files
            ],
        }

    def to_json(self, indent_size: int = 2) -> str:
        """Serialise manifest to pretty-printed JSON."""
        return json.dumps(self.to_dict(), indent=indent_size, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """
    Final result returned by ``ProjectExporter.export()``.

    Includes success flag, manifest, and any errors encountered.
    """

    success: bool
    manifest: ExportManifest
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Directory structure specification
# ---------------------------------------------------------------------------

# Directories to create under the output root (relative paths)
_PROJECT_DIRECTORIES: Tuple[str, ...] = (
    "",
    "models",
    "schemas",
    "routers",
    "core",
    "alembic",
    "alembic/versions",
    "tests",
)


# ---------------------------------------------------------------------------
# ProjectExporter class
# ---------------------------------------------------------------------------


class ProjectExporter:
    """
    Manages writing generated code files to the filesystem.

    Usage::

        exporter = ProjectExporter(config, output_dir=Path("./output"))
        result = exporter.export(generated_files)
        print(result.manifest.to_json())

    Thread-safety: NOT thread-safe.  Use one exporter per output directory.
    """

    def __init__(
        self,
        config: GenerationConfig,
        output_dir: Path,
        *,
        clean_before_export: bool = False,
        atomic_writes: bool = True,
        generate_manifest: bool = True,
        generate_project_files: bool = True,
    ) -> None:
        """
        Initialise the exporter.

        Args:
            config: Generation configuration.
            output_dir: Root directory for output files.
            clean_before_export: If True, wipe the output directory first.
            atomic_writes: If True, use write-to-temp+rename pattern.
            generate_manifest: If True, write a manifest.json file.
            generate_project_files: If True, write pyproject.toml, .env, etc.
        """
        self._config: GenerationConfig = config
        self._output_dir: Path = output_dir.resolve()
        self._clean_before_export: bool = clean_before_export
        self._atomic_writes: bool = atomic_writes
        self._generate_manifest: bool = generate_manifest
        self._generate_project_files: bool = generate_project_files

        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._file_records: List[FileRecord] = []

        logger.debug(
            "ProjectExporter initialised: output_dir=%s, atomic=%s.",
            self._output_dir,
            self._atomic_writes,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def export(
        self,
        generated_files: Dict[str, str],
        schema: Optional[SchemaDefinition] = None,
    ) -> ExportResult:
        """
        Export all generated code files to the filesystem.

        Args:
            generated_files: Mapping of relative_path → file_content.
            schema: Optional schema definition for generating extras.

        Returns:
            ExportResult with success flag, manifest, and error details.
        """
        with Timer("export") as timer:
            try:
                self._pre_export_cleanup()
                self._create_directory_structure()
                self._write_generated_files(generated_files)

                if self._generate_project_files:
                    self._write_project_support_files(schema)

                if self._generate_manifest:
                    self._write_manifest_file()

            except Exception as exc:
                error_msg: str = f"Fatal export error: {type(exc).__name__}: {exc}"
                self._errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

        manifest: ExportManifest = self._build_manifest()
        success: bool = len(self._errors) == 0

        result: ExportResult = ExportResult(
            success=success,
            manifest=manifest,
            errors=tuple(self._errors),
            warnings=tuple(self._warnings),
            elapsed_seconds=timer.elapsed,
        )

        if success:
            logger.info(
                "Export completed successfully: %d files, %d bytes, %.3fs.",
                manifest.total_files,
                manifest.total_bytes,
                timer.elapsed,
            )
        else:
            logger.error(
                "Export completed with %d error(s) in %.3fs.",
                len(self._errors),
                timer.elapsed,
            )

        return result

    # -----------------------------------------------------------------
    # Internal: directory management
    # -----------------------------------------------------------------

    def _pre_export_cleanup(self) -> None:
        """Clean output directory if configured to do so."""
        if not self._clean_before_export:
            return

        if self._output_dir.exists():
            logger.info(
                "Cleaning output directory: %s", self._output_dir
            )
            for item in self._output_dir.iterdir():
                if item.name in {".git", ".gitignore", ".gitkeep"}:
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except OSError as exc:
                    warning_msg: str = (
                        f"Could not remove {item}: {exc}"
                    )
                    self._warnings.append(warning_msg)
                    logger.warning(warning_msg)

    def _create_directory_structure(self) -> None:
        """Create all required directories under the output root."""
        pkg_root: Path = self._output_dir / self._config.package_name

        for rel_dir in _PROJECT_DIRECTORIES:
            dir_path: Path = pkg_root / rel_dir if rel_dir else pkg_root
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug("Created directory: %s", dir_path)
            except OSError as exc:
                error_msg: str = (
                    f"Failed to create directory {dir_path}: {exc}"
                )
                self._errors.append(error_msg)
                logger.error(error_msg)

        # Also create top-level directories outside the package
        for extra_dir in ("tests",):
            extra_path: Path = self._output_dir / extra_dir
            try:
                extra_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._warnings.append(
                    f"Could not create {extra_path}: {exc}"
                )

        logger.info(
            "Directory structure created under: %s", pkg_root
        )

    # -----------------------------------------------------------------
    # Internal: file writing
    # -----------------------------------------------------------------

    def _write_generated_files(
        self, generated_files: Dict[str, str]
    ) -> None:
        """Write all generated code files to disk."""
        pkg_root: Path = self._output_dir / self._config.package_name

        for rel_path, content in generated_files.items():
            full_path: Path = pkg_root / rel_path

            try:
                record: FileRecord = self._write_single_file(
                    full_path, content, rel_path
                )
                self._file_records.append(record)
            except Exception as exc:
                error_msg: str = (
                    f"Failed to write {rel_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._errors.append(error_msg)
                logger.error(error_msg)

        logger.info(
            "Wrote %d generated files to %s.",
            len(self._file_records),
            pkg_root,
        )

    def _write_single_file(
        self,
        full_path: Path,
        content: str,
        rel_path: str,
    ) -> FileRecord:
        """
        Write a single file atomically and return a FileRecord.

        Atomic strategy:
            1. Write to a temporary file in the same directory.
            2. Compute checksum of the content.
            3. Rename temp file to target path (atomic on POSIX).
            4. If rename fails, fall back to direct write.
        """
        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        encoded: bytes = content.encode("utf-8")
        size_bytes: int = len(encoded)
        line_count: int = count_lines(content)
        checksum: str = sha256_hex(content)

        if self._atomic_writes:
            self._atomic_write(full_path, encoded)
        else:
            full_path.write_bytes(encoded)

        logger.debug(
            "Wrote file: %s (%d bytes, %d lines).",
            rel_path,
            size_bytes,
            line_count,
        )

        return FileRecord(
            relative_path=rel_path,
            absolute_path=str(full_path),
            size_bytes=size_bytes,
            line_count=line_count,
            sha256=checksum,
        )

    @staticmethod
    def _atomic_write(target_path: Path, data: bytes) -> None:
        """
        Write data to target_path atomically using a temporary file.

        On POSIX systems, os.rename() is atomic if source and destination
        are on the same filesystem.  We create the temp file in the same
        directory to guarantee this.
        """
        fd: int = -1
        tmp_path: str = ""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target_path.parent),
                prefix=f".{target_path.name}.",
                suffix=".tmp",
            )
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            fd = -1  # Mark as closed

            # Atomic rename
            os.replace(tmp_path, str(target_path))

        except Exception:
            # Close FD if still open
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

            # Clean up temp file if it exists
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # Fall back to direct write
            target_path.write_bytes(data)

    # -----------------------------------------------------------------
    # Internal: project support files
    # -----------------------------------------------------------------

    def _write_project_support_files(
        self,
        schema: Optional[SchemaDefinition] = None,
    ) -> None:
        """Write non-generated project support files."""
        support_files: Dict[str, str] = {}

        # pyproject.toml
        support_files["pyproject.toml"] = self._generate_pyproject_toml()

        # .env file
        support_files[".env"] = self._generate_dotenv()

        # .env.example
        support_files[".env.example"] = self._generate_dotenv_example()

        # .gitignore
        support_files[".gitignore"] = self._generate_gitignore()

        # README.md
        support_files["README.md"] = self._generate_readme(schema)

        # requirements.txt
        support_files["requirements.txt"] = self._generate_requirements()

        # Alembic ini
        if self._config.generate_alembic:
            support_files["alembic.ini"] = self._generate_alembic_ini()

        # docker-compose.yml (optional convenience)
        support_files["docker-compose.yml"] = self._generate_docker_compose()

        # Dockerfile
        support_files["Dockerfile"] = self._generate_dockerfile()

        # tests/__init__.py
        support_files["tests/__init__.py"] = (
            '"""Test suite for the generated API."""\n'
        )

        # tests/conftest.py
        support_files["tests/conftest.py"] = self._generate_conftest()

        for rel_path, content in support_files.items():
            full_path: Path = self._output_dir / rel_path
            try:
                record: FileRecord = self._write_single_file(
                    full_path, content, rel_path
                )
                self._file_records.append(record)
            except Exception as exc:
                warning_msg: str = (
                    f"Could not write support file {rel_path}: {exc}"
                )
                self._warnings.append(warning_msg)
                logger.warning(warning_msg)

        logger.info(
            "Wrote %d project support files.",
            len(support_files),
        )

    def _generate_pyproject_toml(self) -> str:
        """Generate a modern pyproject.toml."""
        lines: List[str] = []
        lines.append("[build-system]")
        lines.append('requires = ["setuptools>=68.0", "wheel"]')
        lines.append('build-backend = "setuptools.build_meta"')
        lines.append("")
        lines.append("[project]")
        lines.append(f'name = "{self._config.project_name}"')
        lines.append(f'version = "{self._config.project_version}"')
        lines.append(
            f'description = "{self._config.project_description}"'
        )
        lines.append('requires-python = ">=3.10"')
        lines.append("dependencies = [")
        lines.append('    "fastapi>=0.109.0",')
        lines.append('    "uvicorn[standard]>=0.27.0",')
        lines.append('    "sqlalchemy>=2.0.25",')
        lines.append('    "pydantic>=2.6.0",')
        lines.append('    "pydantic-settings>=2.1.0",')
        lines.append('    "alembic>=1.13.0",')

        if self._config.use_async:
            if self._config.database_dialect == "postgresql":
                lines.append('    "asyncpg>=0.29.0",')
            elif self._config.database_dialect == "mysql":
                lines.append('    "aiomysql>=0.2.0",')
            elif self._config.database_dialect == "sqlite":
                lines.append('    "aiosqlite>=0.19.0",')
        else:
            if self._config.database_dialect == "postgresql":
                lines.append('    "psycopg2-binary>=2.9.9",')
            elif self._config.database_dialect == "mysql":
                lines.append('    "pymysql>=1.1.0",')

        lines.append('    "python-dotenv>=1.0.0",')
        lines.append("]")
        lines.append("")
        lines.append("[project.optional-dependencies]")
        lines.append("dev = [")
        lines.append('    "pytest>=8.0.0",')
        lines.append('    "pytest-asyncio>=0.23.0",')
        lines.append('    "httpx>=0.27.0",')
        lines.append('    "ruff>=0.2.0",')
        lines.append('    "mypy>=1.8.0",')
        lines.append("]")
        lines.append("")
        lines.append("[tool.ruff]")
        lines.append("line-length = 99")
        lines.append('target-version = "py310"')
        lines.append("")
        lines.append("[tool.ruff.lint]")
        lines.append('select = ["E", "F", "W", "I", "N", "UP"]')
        lines.append("")
        lines.append("[tool.mypy]")
        lines.append("python_version = \"3.10\"")
        lines.append("strict = true")
        lines.append("warn_return_any = true")
        lines.append("warn_unused_configs = true")
        lines.append("")
        lines.append("[tool.pytest.ini_options]")
        lines.append("asyncio_mode = \"auto\"")
        lines.append('testpaths = ["tests"]')
        lines.append("")

        return "\n".join(lines)

    def _generate_dotenv(self) -> str:
        """Generate .env file with default values."""
        lines: List[str] = []
        lines.append("# NexaFlow APIGen — Environment Variables")
        lines.append(
            f"# Generated for project: {self._config.project_name}"
        )
        lines.append("")
        lines.append(
            f'DATABASE_URL="{self._config.database_url}"'
        )
        lines.append("")
        lines.append("# Application settings")
        lines.append('APP_HOST="0.0.0.0"')
        lines.append("APP_PORT=8000")
        lines.append('APP_ENV="development"')
        lines.append('APP_DEBUG="true"')
        lines.append('APP_LOG_LEVEL="INFO"')
        lines.append("")
        lines.append("# Security")
        lines.append(
            'SECRET_KEY="CHANGE-ME-TO-A-STRONG-RANDOM-SECRET"'
        )
        lines.append("")
        lines.append("# CORS")
        cors_str: str = ",".join(self._config.cors_origins)
        lines.append(f'CORS_ORIGINS="{cors_str}"')
        lines.append("")

        return "\n".join(lines)

    def _generate_dotenv_example(self) -> str:
        """Generate .env.example with placeholder values."""
        lines: List[str] = []
        lines.append("# Copy this file to .env and fill in values")
        lines.append("")
        lines.append("DATABASE_URL=")
        lines.append("")
        lines.append('APP_HOST="0.0.0.0"')
        lines.append("APP_PORT=8000")
        lines.append('APP_ENV="development"')
        lines.append('APP_DEBUG="false"')
        lines.append('APP_LOG_LEVEL="INFO"')
        lines.append("")
        lines.append("SECRET_KEY=")
        lines.append("")
        lines.append('CORS_ORIGINS="http://localhost:3000"')
        lines.append("")

        return "\n".join(lines)

    def _generate_gitignore(self) -> str:
        """Generate .gitignore for a Python project."""
        lines: List[str] = []
        lines.append("# Byte-compiled / optimised / DLL files")
        lines.append("__pycache__/")
        lines.append("*.py[cod]")
        lines.append("*$py.class")
        lines.append("")
        lines.append("# Virtual environments")
        lines.append("venv/")
        lines.append(".venv/")
        lines.append("env/")
        lines.append("")
        lines.append("# IDE")
        lines.append(".idea/")
        lines.append(".vscode/")
        lines.append("*.swp")
        lines.append("*.swo")
        lines.append("")
        lines.append("# Environment variables")
        lines.append(".env")
        lines.append("")
        lines.append("# Distribution / packaging")
        lines.append("dist/")
        lines.append("build/")
        lines.append("*.egg-info/")
        lines.append("")
        lines.append("# Testing")
        lines.append(".pytest_cache/")
        lines.append("htmlcov/")
        lines.append(".coverage")
        lines.append("coverage.xml")
        lines.append("")
        lines.append("# mypy")
        lines.append(".mypy_cache/")
        lines.append("")
        lines.append("# ruff")
        lines.append(".ruff_cache/")
        lines.append("")
        lines.append("# Alembic")
        lines.append("alembic/versions/*.pyc")
        lines.append("")
        lines.append("# OS files")
        lines.append(".DS_Store")
        lines.append("Thumbs.db")
        lines.append("")

        return "\n".join(lines)

    def _generate_readme(
        self, schema: Optional[SchemaDefinition] = None
    ) -> str:
        """Generate a project README.md."""
        lines: List[str] = []
        lines.append(f"# {self._config.project_name}")
        lines.append("")
        lines.append(f"> {self._config.project_description}")
        lines.append("")
        lines.append(
            "**Auto-generated by [NexaFlow APIGen](https://github.com/nexaflow/apigen)**"
        )
        lines.append("")
        lines.append("## Quick Start")
        lines.append("")
        lines.append("
```bash")
lines.append("# 1. Create a virtual environment")
lines.append("python -m venv venv")
lines.append("source venv/bin/activate  # Linux/macOS")
lines.append("# venv\\Scripts\\activate  # Windows")
lines.append("")
lines.append("# 2. Install dependencies")
lines.append("pip install -e '.[dev]'")
lines.append("")
lines.append("# 3. Configure environment")
lines.append("cp .env.example .env")
lines.append("# Edit .env with your database URL")
lines.append("")
lines.append("# 4. Run database migrations")
lines.append("alembic upgrade head")
lines.append("")
lines.append("# 5. Start the server")
pkg_name: str = self._config.package_name
lines.append(
f"uvicorn {pkg_name}.main:app --reload --host 0.0.0.0 --port 8000"
)
lines.append("
```")
        lines.append("")
        lines.append("## API Documentation")
        lines.append("")
        lines.append(
            "Once running, visit:"
        )
        lines.append("")
        lines.append(
            "- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)"
        )
        lines.append(
            "- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)"
        )
        lines.append(
            "- **OpenAPI JSON:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)"
        )
        lines.append("")

        if schema and schema.tables:
            lines.append("## Database Tables")
            lines.append("")
            lines.append("| Table | Columns | Endpoints |")
            lines.append("|-------|---------|-----------|")
            for table in schema.tables:
                col_count: int = len(table.columns)
                prefix: str = f"/{to_snake_case(table.name)}"
                lines.append(
                    f"| `{table.name}` | {col_count} | "
                    f"`{self._config.api_prefix}{prefix}` |"
                )
            lines.append("")

        lines.append("## Project Structure")
        lines.append("")
        lines.append("
```")
lines.append(f"{self._config.project_name}/")
lines.append(f"├── {pkg_name}/")
lines.append(f"│   ├── __init__.py")
lines.append(f"│   ├── main.py           # FastAPI application")
lines.append(f"│   ├── database.py        # Engine & session")
lines.append(f"│   ├── models/            # SQLAlchemy ORM models")
lines.append(f"│   ├── schemas/           # Pydantic schemas")
lines.append(f"│   ├── routers/           # API route handlers")
lines.append(f"│   └── core/              # Config & utilities")
lines.append(f"├── alembic/               # Database migrations")
lines.append(f"├── tests/                 # Test suite")
lines.append(f"├── pyproject.toml")
lines.append(f"├── .env")
lines.append(f"└── README.md")
lines.append("
```")
        lines.append("")
        lines.append("## Testing")
        lines.append("")
        lines.append("
```bash")
lines.append("pytest -v")
lines.append("
```")
        lines.append("")
        lines.append("## License")
        lines.append("")
        lines.append(f"This project was auto-generated by NexaFlow APIGen v1.0.0.")
        lines.append("")

        return "\n".join(lines)

    def _generate_requirements(self) -> str:
        """Generate a flat requirements.txt."""
        lines: List[str] = []
        lines.append("# Auto-generated by NexaFlow APIGen")
        lines.append("fastapi>=0.109.0")
        lines.append("uvicorn[standard]>=0.27.0")
        lines.append("sqlalchemy>=2.0.25")
        lines.append("pydantic>=2.6.0")
        lines.append("pydantic-settings>=2.1.0")
        lines.append("alembic>=1.13.0")
        lines.append("python-dotenv>=1.0.0")

        if self._config.use_async:
            if self._config.database_dialect == "postgresql":
                lines.append("asyncpg>=0.29.0")
            elif self._config.database_dialect == "mysql":
                lines.append("aiomysql>=0.2.0")
            elif self._config.database_dialect == "sqlite":
                lines.append("aiosqlite>=0.19.0")
        else:
            if self._config.database_dialect == "postgresql":
                lines.append("psycopg2-binary>=2.9.9")
            elif self._config.database_dialect == "mysql":
                lines.append("pymysql>=1.1.0")

        lines.append("")
        return "\n".join(lines)

    def _generate_alembic_ini(self) -> str:
        """Generate alembic.ini configuration."""
        lines: List[str] = []
        lines.append("[alembic]")
        lines.append(
            f"script_location = {self._config.package_name}/alembic"
        )
        lines.append("prepend_sys_path = .")
        lines.append("")
        lines.append(
            f"sqlalchemy.url = {self._config.database_url}"
        )
        lines.append("")
        lines.append("[post_write_hooks]")
        lines.append("")
        lines.append("[loggers]")
        lines.append("keys = root,sqlalchemy,alembic")
        lines.append("")
        lines.append("[handlers]")
        lines.append("keys = console")
        lines.append("")
        lines.append("[formatters]")
        lines.append("keys = generic")
        lines.append("")
        lines.append("[logger_root]")
        lines.append("level = WARNING")
        lines.append("handlers = console")
        lines.append("")
        lines.append("[logger_sqlalchemy]")
        lines.append("level = WARNING")
        lines.append("handlers =")
        lines.append("qualname = sqlalchemy.engine")
        lines.append("")
        lines.append("[logger_alembic]")
        lines.append("level = INFO")
        lines.append("handlers =")
        lines.append("qualname = alembic")
        lines.append("")
        lines.append("[handler_console]")
        lines.append("class = StreamHandler")
        lines.append("args = (sys.stderr,)")
        lines.append("level = NOTSET")
        lines.append("formatter = generic")
        lines.append("")
        lines.append("[formatter_generic]")
        lines.append(
            "format = %%(levelname)-5.5s [%%(name)s] %%(message)s"
        )
        lines.append("datefmt = %%H:%%M:%%S")
        lines.append("")

        return "\n".join(lines)

    def _generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for development."""
        lines: List[str] = []
        lines.append("# Auto-generated by NexaFlow APIGen")
        lines.append("version: '3.8'")
        lines.append("")
        lines.append("services:")
        lines.append("  app:")
        lines.append(f"    build: .")
        lines.append("    ports:")
        lines.append('      - "8000:8000"')
        lines.append("    env_file:")
        lines.append("      - .env")
        lines.append("    depends_on:")

        if self._config.database_dialect == "postgresql":
            lines.append("      - postgres")
            lines.append("    volumes:")
            lines.append("      - .:/app")
            lines.append("")
            lines.append("  postgres:")
            lines.append("    image: postgres:16-alpine")
            lines.append("    environment:")
            lines.append("      POSTGRES_USER: postgres")
            lines.append("      POSTGRES_PASSWORD: postgres")
            lines.append(
                f"      POSTGRES_DB: {self._config.project_name}"
            )
            lines.append("    ports:")
            lines.append('      - "5432:5432"')
            lines.append("    volumes:")
            lines.append("      - pgdata:/var/lib/postgresql/data")
            lines.append("")
            lines.append("volumes:")
            lines.append("  pgdata:")
        elif self._config.database_dialect == "mysql":
            lines.append("      - mysql")
            lines.append("    volumes:")
            lines.append("      - .:/app")
            lines.append("")
            lines.append("  mysql:")
            lines.append("    image: mysql:8.0")
            lines.append("    environment:")
            lines.append("      MYSQL_ROOT_PASSWORD: root")
            lines.append(
                f"      MYSQL_DATABASE: {self._config.project_name}"
            )
            lines.append("    ports:")
            lines.append('      - "3306:3306"')
            lines.append("    volumes:")
            lines.append("      - mysqldata:/var/lib/mysql")
            lines.append("")
            lines.append("volumes:")
            lines.append("  mysqldata:")
        else:
            # SQLite — no external service needed
            lines.append("      []")
            lines.append("    volumes:")
            lines.append("      - .:/app")

        lines.append("")
        return "\n".join(lines)

    def _generate_dockerfile(self) -> str:
        """Generate a production-ready Dockerfile."""
        lines: List[str] = []
        lines.append("# Auto-generated by NexaFlow APIGen")
        lines.append("FROM python:3.12-slim AS base")
        lines.append("")
        lines.append("WORKDIR /app")
        lines.append("")
        lines.append("# Install system dependencies")
        lines.append(
            "RUN apt-get update && apt-get install -y --no-install-recommends \\"
        )
        lines.append("    gcc libpq-dev \\")
        lines.append("    && rm -rf /var/lib/apt/lists/*")
        lines.append("")
        lines.append("# Copy requirements first for layer caching")
        lines.append("COPY requirements.txt .")
        lines.append(
            "RUN pip install --no-cache-dir -r requirements.txt"
        )
        lines.append("")
        lines.append("# Copy application code")
        lines.append("COPY . .")
        lines.append("")
        lines.append("# Install the project")
        lines.append("RUN pip install --no-cache-dir -e .")
        lines.append("")
        lines.append("# Expose port")
        lines.append("EXPOSE 8000")
        lines.append("")
        lines.append("# Health check")
        lines.append(
            "HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\"
        )
        lines.append(
            "    CMD python -c "
            "\"import urllib.request; "
            "urllib.request.urlopen('http://localhost:8000/health')\""
        )
        lines.append("")
        lines.append("# Run the application")
        pkg_name: str = self._config.package_name
        lines.append(
            f'CMD ["uvicorn", "{pkg_name}.main:app", '
            f'"--host", "0.0.0.0", "--port", "8000"]'
        )
        lines.append("")

        return "\n".join(lines)

    def _generate_conftest(self) -> str:
        """Generate tests/conftest.py with test fixtures."""
        lines: List[str] = []
        pkg_name: str = self._config.package_name
        lines.append('"""')
        lines.append("Pytest fixtures for API testing.")
        lines.append("Auto-generated by NexaFlow APIGen.")
        lines.append('"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append("import asyncio")
        lines.append("from collections.abc import AsyncGenerator")
        lines.append("from typing import Generator")
        lines.append("")
        lines.append("import pytest")
        lines.append("import pytest_asyncio")
        lines.append("from httpx import ASGITransport, AsyncClient")

        if self._config.use_async:
            lines.append(
                "from sqlalchemy.ext.asyncio import ("
            )
            lines.append("    AsyncSession,")
            lines.append("    async_sessionmaker,")
            lines.append("    create_async_engine,")
            lines.append(")")
        else:
            lines.append("from sqlalchemy import create_engine")
            lines.append(
                "from sqlalchemy.orm import Session, sessionmaker"
            )

        lines.append("")
        lines.append(f"from {pkg_name}.database import Base, get_db")
        lines.append(f"from {pkg_name}.main import app")
        lines.append("")
        lines.append("")
        lines.append(
            "# Use an in-memory SQLite database for tests"
        )

        if self._config.use_async:
            lines.append(
                'TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"'
            )
            lines.append("")
            lines.append(
                "test_engine = create_async_engine("
                "TEST_DATABASE_URL, echo=False)"
            )
            lines.append(
                "TestSessionLocal = async_sessionmaker("
            )
            lines.append("    bind=test_engine,")
            lines.append("    class_=AsyncSession,")
            lines.append("    expire_on_commit=False,")
            lines.append(")")
            lines.append("")
            lines.append("")
            lines.append(
                "async def override_get_db() -> "
                "AsyncGenerator[AsyncSession, None]:"
            )
            lines.append(
                "    async with TestSessionLocal() as session:"
            )
            lines.append("        yield session")
            lines.append("")
            lines.append("")
            lines.append("app.dependency_overrides[get_db] = override_get_db")
            lines.append("")
            lines.append("")
            lines.append("@pytest_asyncio.fixture(autouse=True)")
            lines.append(
                "async def setup_database() -> AsyncGenerator[None, None]:"
            )
            lines.append('    """Create tables before each test, drop after."""')
            lines.append(
                "    async with test_engine.begin() as conn:"
            )
            lines.append(
                "        await conn.run_sync(Base.metadata.create_all)"
            )
            lines.append("    yield")
            lines.append(
                "    async with test_engine.begin() as conn:"
            )
            lines.append(
                "        await conn.run_sync(Base.metadata.drop_all)"
            )
            lines.append("")
            lines.append("")
            lines.append("@pytest_asyncio.fixture")
            lines.append(
                "async def client() -> AsyncGenerator[AsyncClient, None]:"
            )
            lines.append('    """Async HTTP client for testing."""')
            lines.append(
                "    transport = ASGITransport(app=app)"
            )
            lines.append(
                "    async with AsyncClient("
            )
            lines.append(
                '        transport=transport, base_url="http://test"'
            )
            lines.append("    ) as ac:")
            lines.append("        yield ac")
        else:
            lines.append(
                'TEST_DATABASE_URL = "sqlite:///:memory:"'
            )
            lines.append("")
            lines.append(
                "test_engine = create_engine("
                "TEST_DATABASE_URL, echo=False)"
            )
            lines.append(
                "TestSessionLocal = sessionmaker("
            )
            lines.append("    bind=test_engine,")
            lines.append("    autocommit=False,")
            lines.append("    autoflush=False,")
            lines.append(")")
            lines.append("")
            lines.append("")
            lines.append(
                "def override_get_db() -> Generator[Session, None, None]:"
            )
            lines.append(
                "    db = TestSessionLocal()"
            )
            lines.append("    try:")
            lines.append("        yield db")
            lines.append("    finally:")
            lines.append("        db.close()")
            lines.append("")
            lines.append("")
            lines.append("app.dependency_overrides[get_db] = override_get_db")
            lines.append("")
            lines.append("")
            lines.append("@pytest.fixture(autouse=True)")
            lines.append(
                "def setup_database() -> Generator[None, None, None]:"
            )
            lines.append('    """Create tables before each test, drop after."""')
            lines.append(
                "    Base.metadata.create_all(bind=test_engine)"
            )
            lines.append("    yield")
            lines.append(
                "    Base.metadata.drop_all(bind=test_engine)"
            )

        lines.append("")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Internal: manifest
    # -----------------------------------------------------------------

    def _build_manifest(self) -> ExportManifest:
        """Build the export manifest from collected file records."""
        import apigen

        total_bytes: int = sum(r.size_bytes for r in self._file_records)
        total_lines: int = sum(r.line_count for r in self._file_records)

        return ExportManifest(
            project_name=self._config.project_name,
            project_version=self._config.project_version,
            generator_version=apigen.__version__,
            export_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            output_directory=str(self._output_dir),
            total_files=len(self._file_records),
            total_bytes=total_bytes,
            total_lines=total_lines,
            files=list(self._file_records),
        )

    def _write_manifest_file(self) -> None:
        """Write the manifest.json to the output directory."""
        manifest: ExportManifest = self._build_manifest()
        manifest_path: Path = self._output_dir / "manifest.json"

        try:
            content: str = manifest.to_json()
            record: FileRecord = self._write_single_file(
                manifest_path, content, "manifest.json"
            )
            self._file_records.append(record)
            logger.debug("Wrote manifest to %s.", manifest_path)
        except Exception as exc:
            self._warnings.append(
                f"Could not write manifest: {exc}"
            )
            logger.warning("Failed to write manifest: %s", exc)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "ProjectExporter",
    "ExportManifest",
    "ExportResult",
    "FileRecord",
]

logger.debug("apigen.exporters loaded.")
