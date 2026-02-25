# File: apigen/templates.py
"""
NexaFlow APIGen - Code Template Engine
========================================
Pure-Python, zero-dependency code generation engine.

This module is the **heart** of NexaFlow — it transforms ``TableInfo`` and
``GenerationConfig`` objects into production-ready Python source code strings
for:
    1. SQLAlchemy 2.0 ORM models (using ``Mapped[]`` / ``mapped_column()``)
    2. Pydantic V2 CRUD schemas (Create / Read / Update / List)
    3. FastAPI routers with full CRUD endpoints
    4. Database session & engine configuration
    5. Alembic migration environment
    6. Application entry point (main.py)

**Performance contract:**
    - All string assembly uses ``List[str]`` + ``"\\n".join()`` pattern.
    - No ``str += str`` concatenation anywhere.
    - Template methods are stateless — safe for concurrent use.

**Quality contract:**
    - Generated code is PEP-8 compliant (99-char line length).
    - Generated code uses modern Python 3.10+ / SQLAlchemy 2.0 idioms.
    - Every generated file is self-contained and importable.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from apigen.models import (
    AuthStrategy,
    ColumnInfo,
    ColumnType,
    CRUDConfig,
    DatabaseDialect,
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
from apigen.utils import (
    build_import_block,
    collect_column_imports,
    column_to_field_name,
    indent,
    indent_lines,
    make_docstring,
    merge_import_dicts,
    safe_identifier,
    table_to_class_name,
    table_to_router_prefix,
    table_to_router_tag,
    to_camel_case,
    to_pascal_case,
    to_plural,
    to_singular,
    to_snake_case,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen.templates")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDENT: str = "    "  # 4-space indent
_DOUBLE_INDENT: str = "        "
_TRIPLE_INDENT: str = "            "

# Mapping from ColumnType enum to Mapped[] Python type annotations
_MAPPED_TYPE_MAP: Dict[str, str] = {
    "integer": "int",
    "biginteger": "int",
    "smallinteger": "int",
    "float": "float",
    "numeric": "Decimal",
    "double": "float",
    "boolean": "bool",
    "string": "str",
    "text": "str",
    "varchar": "str",
    "char": "str",
    "binary": "bytes",
    "largebinary": "bytes",
    "date": "date",
    "datetime": "datetime",
    "time": "time",
    "timestamp": "datetime",
    "interval": "timedelta",
    "uuid": "UUID",
    "json": "Dict[str, Any]",
    "jsonb": "Dict[str, Any]",
    "array": "List[Any]",
    "enum": "str",
    "hstore": "Dict[str, str]",
    "inet": "str",
    "cidr": "str",
    "macaddr": "str",
}

# Pydantic field type mapping (slightly different from ORM)
_PYDANTIC_TYPE_MAP: Dict[str, str] = {
    "integer": "int",
    "biginteger": "int",
    "smallinteger": "int",
    "float": "float",
    "numeric": "Decimal",
    "double": "float",
    "boolean": "bool",
    "string": "str",
    "text": "str",
    "varchar": "str",
    "char": "str",
    "binary": "bytes",
    "largebinary": "bytes",
    "date": "date",
    "datetime": "datetime",
    "time": "time",
    "timestamp": "datetime",
    "interval": "timedelta",
    "uuid": "UUID",
    "json": "Dict[str, Any]",
    "jsonb": "Dict[str, Any]",
    "array": "List[Any]",
    "enum": "str",
    "hstore": "Dict[str, str]",
    "inet": "str",
    "cidr": "str",
    "macaddr": "str",
}


# ---------------------------------------------------------------------------
# Helper: SQLAlchemy mapped_column() argument builder
# ---------------------------------------------------------------------------


def _build_mapped_column_args(col: ColumnInfo, table: TableInfo) -> str:
    """
    Build the argument string for ``mapped_column(...)``.

    Returns a string like:
        ``Integer, primary_key=True, autoincrement=True``

    Complexity: O(k) where k = number of column attributes (bounded constant).
    """
    parts: List[str] = []

    # Type argument
    parts.append(col.sqlalchemy_type_str)

    # Primary key
    if col.primary_key:
        parts.append("primary_key=True")

    # Autoincrement
    if col.autoincrement:
        parts.append("autoincrement=True")

    # Nullable (only emit if not PK — PKs are implicitly NOT NULL)
    if not col.primary_key:
        if col.nullable:
            parts.append("nullable=True")
        else:
            parts.append("nullable=False")

    # Unique
    if col.unique and not col.primary_key:
        parts.append("unique=True")

    # Index
    if col.index:
        parts.append("index=True")

    # Server default
    if col.server_default:
        parts.append(f'server_default=text("{col.server_default}")')

    # Default (Python-side)
    if col.default and not col.default.is_server_default:
        if col.default.is_clause:
            parts.append(f'default=text("{col.default.value}")')
        elif isinstance(col.default.value, str):
            parts.append(f'default="{col.default.value}"')
        else:
            parts.append(f"default={col.default.value}")

    # Foreign key
    fk_col_names: FrozenSet[str] = table.fk_column_names
    if col.name in fk_col_names:
        # Find the FK info for this column
        for fk in table.foreign_keys:
            if fk.constrained_column == col.name:
                fk_target: str = f"{fk.referred_table}.{fk.referred_column}"
                on_delete_str: str = ""
                if fk.on_delete and fk.on_delete != "NO ACTION":
                    on_delete_str = f', ondelete="{fk.on_delete}"'
                parts.append(
                    f'ForeignKey("{fk_target}"{on_delete_str})'
                )
                break

    # Comment
    if col.comment:
        escaped_comment: str = col.comment.replace('"', '\\"')
        parts.append(f'comment="{escaped_comment}"')

    return ", ".join(parts)


def _get_mapped_type_annotation(col: ColumnInfo) -> str:
    """
    Return the ``Mapped[X]`` type annotation string for a column.

    Examples:
        - PK int → ``Mapped[int]``
        - nullable string → ``Mapped[Optional[str]]``
    """
    base_type: str = _MAPPED_TYPE_MAP.get(col.column_type, "str")

    if col.nullable and not col.primary_key:
        return f"Mapped[Optional[{base_type}]]"
    return f"Mapped[{base_type}]"


# ---------------------------------------------------------------------------
# TemplateGenerator class
# ---------------------------------------------------------------------------


class TemplateGenerator:
    """
    Stateless code-generation engine.

    Accepts ``TableInfo`` and ``GenerationConfig`` instances and produces
    Python source code strings.  Each ``generate_*`` method returns a
    complete, self-contained file content string.

    Thread-safe: no mutable instance state.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self._config: GenerationConfig = config
        self._indent: str = " " * config.indent_size
        self._double_indent: str = self._indent * 2
        self._triple_indent: str = self._indent * 3
        logger.debug(
            "TemplateGenerator initialised (dialect=%s, async=%s).",
            config.database_dialect,
            config.use_async,
        )

    # ===================================================================
    # 1. SQLAlchemy ORM Model
    # ===================================================================

    def generate_orm_model(self, table: TableInfo) -> str:
        """
        Generate a complete SQLAlchemy 2.0 ORM model file for one table.

        Output uses ``DeclarativeBase``, ``Mapped[]``, ``mapped_column()``.
        """
        lines: List[str] = []
        class_name: str = table_to_class_name(table.name)

        # --- Collect imports ---
        imports: Dict[str, Set[str]] = {
            "sqlalchemy": {
                "ForeignKey", "String", "Text", "text",
            },
            "sqlalchemy.orm": {
                "Mapped", "mapped_column", "relationship",
            },
        }

        # Add type-specific SA imports
        for col in table.columns:
            sa_base: str = col.sqlalchemy_type_str.split("(")[0].strip()
            if sa_base == "SQLEnum":
                imports.setdefault("sqlalchemy", set()).add("Enum as SQLEnum")
            elif sa_base in ("UUID", "JSONB", "ARRAY", "HSTORE", "INET", "CIDR", "MACADDR"):
                imports.setdefault("sqlalchemy.dialects.postgresql", set()).add(sa_base)
            else:
                imports.setdefault("sqlalchemy", set()).add(sa_base)

        # Python type imports for Mapped[] annotations
        python_imports: Dict[str, Set[str]] = {
            "typing": {"Optional", "List", "Dict", "Any"},
        }
        for col in table.columns:
            base_type: str = _MAPPED_TYPE_MAP.get(col.column_type, "str")
            if base_type == "Decimal":
                python_imports.setdefault("decimal", set()).add("Decimal")
            elif base_type == "UUID":
                python_imports.setdefault("uuid", set()).add("UUID")
            elif base_type in ("date", "datetime", "time", "timedelta"):
                python_imports.setdefault("datetime", set()).add(base_type)

        all_imports: Dict[str, Set[str]] = merge_import_dicts(imports, python_imports)

        # Add base model import
        all_imports.setdefault("apigen_generated.database", set()).add("Base")

        # --- File header ---
        lines.append(f'"""')
        lines.append(f"SQLAlchemy ORM model for table: {table.name}")
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append(build_import_block(all_imports))
        lines.append("")
        lines.append("")

        # --- Class definition ---
        lines.append(f"class {class_name}(Base):")
        if self._config.generate_docstrings:
            lines.append(f'{self._indent}"""')
            lines.append(f"{self._indent}ORM model for the '{table.name}' table.")
            if table.comment:
                lines.append(f"{self._indent}")
                lines.append(f"{self._indent}{table.comment}")
            lines.append(f'{self._indent}"""')
            lines.append("")

        # __tablename__
        lines.append(f'{self._indent}__tablename__ = "{table.name}"')

        # __table_args__
        table_args_parts: List[str] = []

        # Multi-column unique constraints
        for uc in table.unique_constraints:
            cols_str: str = ", ".join(f'"{c}"' for c in uc.columns)
            uc_name: str = uc.name or f"uq_{table.name}_{'_'.join(uc.columns)}"
            table_args_parts.append(
                f'UniqueConstraint({cols_str}, name="{uc_name}")'
            )
            all_imports.setdefault("sqlalchemy", set()).add("UniqueConstraint")

        # Multi-column indexes
        for idx in table.indexes:
            cols_str = ", ".join(f'"{c}"' for c in idx.columns)
            unique_str: str = ", unique=True" if idx.unique else ""
            table_args_parts.append(
                f'Index("{idx.name}", {cols_str}{unique_str})'
            )
            all_imports.setdefault("sqlalchemy", set()).add("Index")

        if table_args_parts or table.schema_name:
            args_inner: List[str] = list(table_args_parts)
            dict_parts: List[str] = []
            if table.schema_name:
                dict_parts.append(f'"schema": "{table.schema_name}"')
            if table.extend_existing:
                dict_parts.append('"extend_existing": True')
            if dict_parts:
                args_inner.append("{" + ", ".join(dict_parts) + "}")

            if len(args_inner) == 1 and args_inner[0].startswith("{"):
                lines.append(f"{self._indent}__table_args__ = ({args_inner[0]},)")
            else:
                lines.append(f"{self._indent}__table_args__ = (")
                for arg in args_inner:
                    lines.append(f"{self._double_indent}{arg},")
                lines.append(f"{self._indent})")
        lines.append("")

        # --- Column definitions ---
        lines.append(f"{self._indent}# --- Columns ---")
        for col in table.columns:
            field_name: str = column_to_field_name(col.name)
            type_annotation: str = _get_mapped_type_annotation(col)
            mc_args: str = _build_mapped_column_args(col, table)
            lines.append(
                f"{self._indent}{field_name}: {type_annotation} = "
                f"mapped_column({mc_args})"
            )

        # --- Relationships ---
        if table.relationships:
            lines.append("")
            lines.append(f"{self._indent}# --- Relationships ---")
            for rel in table.relationships:
                rel_line: str = self._build_relationship_line(rel, table)
                lines.append(f"{self._indent}{rel_line}")

        # --- __repr__ ---
        lines.append("")
        pk_cols: List[str] = table.resolved_primary_keys
        repr_fields: List[str] = pk_cols[:3] if pk_cols else [table.columns[0].name]
        repr_parts: str = ", ".join(
            f"{c}={{self.{column_to_field_name(c)}!r}}" for c in repr_fields
        )
        lines.append(f"{self._indent}def __repr__(self) -> str:")
        lines.append(
            f'{self._double_indent}return f"<{class_name} {repr_parts}>"'
        )

        lines.append("")

        content: str = "\n".join(lines)
        logger.debug(
            "Generated ORM model for '%s': %d lines.",
            table.name,
            content.count("\n") + 1,
        )
        return content

    def _build_relationship_line(
        self, rel: RelationshipInfo, table: TableInfo
    ) -> str:
        """Build a single relationship() declaration line."""
        target_class: str = table_to_class_name(rel.target_table)
        parts: List[str] = [f'"{target_class}"']

        if rel.back_populates:
            parts.append(f'back_populates="{rel.back_populates}"')

        if rel.lazy != "select":
            parts.append(f'lazy="{rel.lazy}"')

        if rel.cascade != "save-update, merge":
            parts.append(f'cascade="{rel.cascade}"')

        uselist: bool = rel.resolved_uselist

        if rel.relationship_type == RelationshipType.MANY_TO_MANY:
            if rel.secondary_table:
                parts.append(f'secondary="{rel.secondary_table}"')

        if not uselist:
            parts.append("uselist=False")

        args_str: str = ", ".join(parts)

        if uselist:
            type_hint: str = f'Mapped[List["{target_class}"]]'
        else:
            type_hint = f'Mapped[Optional["{target_class}"]]'

        return f"{rel.name}: {type_hint} = relationship({args_str})"

    # ===================================================================
    # 2. Pydantic CRUD Schemas
    # ===================================================================

    def generate_pydantic_schemas(self, table: TableInfo) -> str:
        """
        Generate Pydantic V2 CRUD schemas for a table:
        - {ClassName}Base         (shared fields)
        - {ClassName}Create       (POST body)
        - {ClassName}Update       (PATCH body — all Optional)
        - {ClassName}Read         (GET response — includes PK + timestamps)
        - {ClassName}List         (paginated list response)
        """
        lines: List[str] = []
        class_name: str = table_to_class_name(table.name)
        crud: CRUDConfig = self._config.get_crud_config(table.name)

        # --- Imports ---
        schema_imports: Dict[str, Set[str]] = {
            "pydantic": {"BaseModel", "ConfigDict", "Field"},
            "typing": {"Optional", "List", "Dict", "Any"},
        }

        for col in table.columns:
            base_type: str = _PYDANTIC_TYPE_MAP.get(col.column_type, "str")
            if base_type == "Decimal":
                schema_imports.setdefault("decimal", set()).add("Decimal")
            elif base_type == "UUID":
                schema_imports.setdefault("uuid", set()).add("UUID")
            elif base_type in ("date", "datetime", "time", "timedelta"):
                schema_imports.setdefault("datetime", set()).add(base_type)

        # --- File header ---
        lines.append(f'"""')
        lines.append(f"Pydantic V2 schemas for table: {table.name}")
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append(build_import_block(schema_imports))
        lines.append("")
        lines.append("")

        # --- Base schema ---
        lines.extend(
            self._generate_base_schema(table, class_name)
        )
        lines.append("")
        lines.append("")

        # --- Create schema ---
        lines.extend(
            self._generate_create_schema(table, class_name)
        )
        lines.append("")
        lines.append("")

        # --- Update schema ---
        lines.extend(
            self._generate_update_schema(table, class_name)
        )
        lines.append("")
        lines.append("")

        # --- Read schema ---
        lines.extend(
            self._generate_read_schema(table, class_name)
        )
        lines.append("")
        lines.append("")

        # --- List schema ---
        lines.extend(
            self._generate_list_schema(table, class_name)
        )
        lines.append("")

        content: str = "\n".join(lines)
        logger.debug(
            "Generated Pydantic schemas for '%s': %d lines.",
            table.name,
            content.count("\n") + 1,
        )
        return content

    def _generate_base_schema(
        self, table: TableInfo, class_name: str
    ) -> List[str]:
        """Generate the Base schema with shared fields."""
        lines: List[str] = []
        lines.append(f"class {class_name}Base(BaseModel):")
        if self._config.generate_docstrings:
            lines.append(f'{self._indent}"""Shared fields for {class_name}."""')
            lines.append("")

        lines.append(
            f"{self._indent}model_config = ConfigDict("
            f"from_attributes=True, "
            f"populate_by_name=True, "
            f"str_strip_whitespace=True"
            f")"
        )
        lines.append("")

        # Non-PK, non-auto fields
        for col in table.columns:
            if col.primary_key and col.autoincrement:
                continue  # skip auto-PK in base
            field_name: str = column_to_field_name(col.name)
            type_str: str = _PYDANTIC_TYPE_MAP.get(col.column_type, "str")
            field_args: str = self._build_pydantic_field_args(col)

            if col.nullable and not col.primary_key:
                lines.append(
                    f"{self._indent}{field_name}: Optional[{type_str}]"
                    f" = {field_args}"
                )
            else:
                lines.append(
                    f"{self._indent}{field_name}: {type_str}"
                    f" = {field_args}"
                )

        return lines

    def _generate_create_schema(
        self, table: TableInfo, class_name: str
    ) -> List[str]:
        """Generate Create schema (inherits from Base)."""
        lines: List[str] = []
        lines.append(f"class {class_name}Create({class_name}Base):")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Schema for creating a new {class_name}."""'
            )
            lines.append("")
        lines.append(f"{self._indent}pass")
        return lines

    def _generate_update_schema(
        self, table: TableInfo, class_name: str
    ) -> List[str]:
        """Generate Update schema (all fields Optional for PATCH)."""
        lines: List[str] = []
        lines.append(f"class {class_name}Update(BaseModel):")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Schema for updating a {class_name}. '
                f'All fields optional."""'
            )
            lines.append("")

        lines.append(
            f"{self._indent}model_config = ConfigDict("
            f"from_attributes=True"
            f")"
        )
        lines.append("")

        for col in table.columns:
            if col.primary_key:
                continue  # never update PK
            field_name: str = column_to_field_name(col.name)
            type_str: str = _PYDANTIC_TYPE_MAP.get(col.column_type, "str")
            lines.append(
                f"{self._indent}{field_name}: Optional[{type_str}]"
                f" = Field(default=None)"
            )

        return lines

    def _generate_read_schema(
        self, table: TableInfo, class_name: str
    ) -> List[str]:
        """Generate Read schema (includes PK + all fields)."""
        lines: List[str] = []
        lines.append(f"class {class_name}Read({class_name}Base):")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Schema for reading a {class_name} '
                f'(includes PK)."""'
            )
            lines.append("")

        lines.append(
            f"{self._indent}model_config = ConfigDict("
            f"from_attributes=True"
            f")"
        )
        lines.append("")

        # Add PK fields that were excluded from Base
        for col in table.columns:
            if col.primary_key and col.autoincrement:
                field_name: str = column_to_field_name(col.name)
                type_str: str = _PYDANTIC_TYPE_MAP.get(col.column_type, "int")
                lines.append(
                    f"{self._indent}{field_name}: {type_str}"
                )

        return lines

    def _generate_list_schema(
        self, table: TableInfo, class_name: str
    ) -> List[str]:
        """Generate List/Pagination response schema."""
        lines: List[str] = []
        lines.append(f"class {class_name}ListResponse(BaseModel):")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Paginated list response for {class_name}."""'
            )
            lines.append("")

        lines.append(
            f"{self._indent}items: List[{class_name}Read]"
            f' = Field(default_factory=list)'
        )
        lines.append(
            f"{self._indent}total: int"
            f' = Field(default=0, description="Total number of records.")'
        )

        if self._config.pagination_style == "offset":
            lines.append(
                f"{self._indent}offset: int"
                f' = Field(default=0, description="Current offset.")'
            )
            lines.append(
                f"{self._indent}limit: int"
                f" = Field(default={self._config.default_page_size},"
                f' description="Page size.")'
            )
        elif self._config.pagination_style == "page_number":
            lines.append(
                f"{self._indent}page: int"
                f' = Field(default=1, description="Current page number.")'
            )
            lines.append(
                f"{self._indent}page_size: int"
                f" = Field(default={self._config.default_page_size},"
                f' description="Items per page.")'
            )
            lines.append(
                f"{self._indent}total_pages: int"
                f' = Field(default=0, description="Total pages.")'
            )
        elif self._config.pagination_style == "cursor":
            lines.append(
                f"{self._indent}next_cursor: Optional[str]"
                f' = Field(default=None, description="Cursor for next page.")'
            )
            lines.append(
                f"{self._indent}prev_cursor: Optional[str]"
                f' = Field(default=None, description="Cursor for previous page.")'
            )

        lines.append(
            f"{self._indent}has_more: bool"
            f' = Field(default=False, description="More records available.")'
        )

        return lines

    def _build_pydantic_field_args(self, col: ColumnInfo) -> str:
        """Build a ``Field(...)`` expression string for a Pydantic field."""
        parts: List[str] = []

        if col.nullable:
            parts.append("default=None")
        elif col.default and not col.default.is_server_default:
            if isinstance(col.default.value, str):
                parts.append(f'default="{col.default.value}"')
            elif col.default.value is not None:
                parts.append(f"default={col.default.value}")
            else:
                parts.append("default=None")

        if col.max_length and col.is_string:
            parts.append(f"max_length={col.max_length}")

        if col.comment:
            escaped: str = col.comment.replace('"', '\\"')
            parts.append(f'description="{escaped}"')
        else:
            human_name: str = col.name.replace("_", " ").title()
            parts.append(f'description="{human_name}"')

        if col.column_type in (
            ColumnType.STRING, ColumnType.VARCHAR, ColumnType.CHAR
        ):
            if col.max_length:
                parts.append(f"min_length=0")

        if not parts:
            return "Field()"

        return f"Field({', '.join(parts)})"

    # ===================================================================
    # 3. FastAPI Router
    # ===================================================================

    def generate_router(self, table: TableInfo) -> str:
        """
        Generate a complete FastAPI router file for one table with
        full CRUD operations based on ``CRUDConfig``.
        """
        lines: List[str] = []
        class_name: str = table_to_class_name(table.name)
        snake_name: str = to_snake_case(table.name)
        plural_name: str = to_plural(snake_name)
        router_prefix: str = table_to_router_prefix(table.name)
        router_tag: str = table_to_router_tag(table.name)
        crud: CRUDConfig = self._config.get_crud_config(table.name)

        pk_cols: List[str] = table.resolved_primary_keys
        pk_col: str = pk_cols[0] if pk_cols else "id"
        pk_info: Optional[ColumnInfo] = table.get_column(pk_col)
        pk_python_type: str = _PYDANTIC_TYPE_MAP.get(
            pk_info.column_type if pk_info else "integer", "int"
        )

        # --- Imports ---
        router_imports: Dict[str, Set[str]] = {
            "fastapi": {"APIRouter", "Depends", "HTTPException", "Query", "Path"},
            "sqlalchemy.ext.asyncio": {"AsyncSession"},
            "sqlalchemy": {"select", "func", "delete", "update"},
            "sqlalchemy.orm": {"selectinload"},
            "typing": {"Optional", "List", "Dict", "Any"},
        }

        schema_module: str = f"apigen_generated.schemas.{snake_name}"
        router_imports[schema_module] = {
            f"{class_name}Create",
            f"{class_name}Read",
            f"{class_name}Update",
            f"{class_name}ListResponse",
        }

        model_module: str = f"apigen_generated.models.{snake_name}"
        router_imports[model_module] = {class_name}
        router_imports["apigen_generated.database"] = {"get_db"}

        # Add PK type imports
        if pk_python_type == "UUID":
            router_imports.setdefault("uuid", set()).add("UUID")

        # --- File header ---
        lines.append(f'"""')
        lines.append(f"FastAPI router for {class_name} CRUD operations.")
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append(build_import_block(router_imports))
        lines.append("")
        lines.append("")

        # --- Router instance ---
        lines.append(
            f'router = APIRouter(prefix="{router_prefix}", '
            f'tags=["{router_tag}"])'
        )
        lines.append("")
        lines.append("")

        # --- CREATE ---
        if crud.create:
            lines.extend(
                self._gen_create_endpoint(
                    table, class_name, snake_name, pk_col, pk_python_type
                )
            )
            lines.append("")
            lines.append("")

        # --- READ ONE ---
        if crud.read_one:
            lines.extend(
                self._gen_read_one_endpoint(
                    table, class_name, snake_name, pk_col, pk_python_type
                )
            )
            lines.append("")
            lines.append("")

        # --- READ MANY (LIST) ---
        if crud.read_many:
            lines.extend(
                self._gen_read_many_endpoint(
                    table, class_name, snake_name, pk_col
                )
            )
            lines.append("")
            lines.append("")

        # --- UPDATE ---
        if crud.update:
            lines.extend(
                self._gen_update_endpoint(
                    table, class_name, snake_name, pk_col, pk_python_type
                )
            )
            lines.append("")
            lines.append("")

        # --- DELETE ---
        if crud.delete:
            lines.extend(
                self._gen_delete_endpoint(
                    table, class_name, snake_name, pk_col, pk_python_type
                )
            )
            lines.append("")
            lines.append("")

        # --- SEARCH ---
        if crud.search:
            lines.extend(
                self._gen_search_endpoint(
                    table, class_name, snake_name
                )
            )
            lines.append("")
            lines.append("")

        # --- BULK CREATE ---
        if crud.bulk_create:
            lines.extend(
                self._gen_bulk_create_endpoint(
                    table, class_name, snake_name
                )
            )
            lines.append("")
            lines.append("")

        # --- BULK DELETE ---
        if crud.bulk_delete:
            lines.extend(
                self._gen_bulk_delete_endpoint(
                    table, class_name, snake_name, pk_col, pk_python_type
                )
            )
            lines.append("")

        content: str = "\n".join(lines)
        logger.debug(
            "Generated router for '%s': %d lines.",
            table.name,
            content.count("\n") + 1,
        )
        return content

    def _gen_create_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
        pk_col: str,
        pk_type: str,
    ) -> List[str]:
        """Generate POST / create endpoint."""
        lines: List[str] = []
        lines.append(f"@router.post(")
        lines.append(f'{self._indent}"/",')
        lines.append(f"{self._indent}response_model={class_name}Read,")
        lines.append(f"{self._indent}status_code=201,")
        lines.append(
            f'{self._indent}summary="Create a new {class_name}",'
        )
        lines.append(f")")
        lines.append(
            f"async def create_{snake_name}("
        )
        lines.append(
            f"{self._indent}payload: {class_name}Create,"
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> {class_name}Read:")
        if self._config.generate_docstrings:
            lines.append(f'{self._indent}"""Create a new {class_name} record."""')

        lines.append(
            f"{self._indent}db_obj = {class_name}("
            f"**payload.model_dump(exclude_unset=True))"
        )
        lines.append(f"{self._indent}db.add(db_obj)")
        lines.append(f"{self._indent}await db.commit()")
        lines.append(f"{self._indent}await db.refresh(db_obj)")
        lines.append(
            f"{self._indent}return {class_name}Read.model_validate(db_obj)"
        )
        return lines

    def _gen_read_one_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
        pk_col: str,
        pk_type: str,
    ) -> List[str]:
        """Generate GET /{id} read-one endpoint."""
        pk_field: str = column_to_field_name(pk_col)
        lines: List[str] = []
        lines.append(f"@router.get(")
        lines.append(f'{self._indent}"/{{{pk_field}}}",')
        lines.append(f"{self._indent}response_model={class_name}Read,")
        lines.append(
            f'{self._indent}summary="Get a {class_name} by {pk_field}",'
        )
        lines.append(f")")
        lines.append(
            f"async def get_{snake_name}("
        )
        lines.append(
            f"{self._indent}{pk_field}: {pk_type} = Path(...),"
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> {class_name}Read:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Retrieve a single {class_name} by primary key."""'
            )

        lines.append(
            f"{self._indent}stmt = select({class_name}).where("
            f"{class_name}.{pk_field} == {pk_field})"
        )
        lines.append(
            f"{self._indent}result = await db.execute(stmt)"
        )
        lines.append(
            f"{self._indent}db_obj = result.scalar_one_or_none()"
        )
        lines.append(f"{self._indent}if db_obj is None:")
        lines.append(
            f"{self._double_indent}raise HTTPException("
            f"status_code=404, "
            f'detail="{class_name} not found.")'
        )
        lines.append(
            f"{self._indent}return {class_name}Read.model_validate(db_obj)"
        )
        return lines

    def _gen_read_many_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
        pk_col: str,
    ) -> List[str]:
        """Generate GET / list endpoint with pagination."""
        plural_name: str = to_plural(snake_name)
        lines: List[str] = []

        lines.append(f"@router.get(")
        lines.append(f'{self._indent}"/",')
        lines.append(
            f"{self._indent}response_model={class_name}ListResponse,"
        )
        lines.append(
            f'{self._indent}summary="List {to_plural(class_name)}",'
        )
        lines.append(f")")
        lines.append(f"async def list_{plural_name}(")

        if self._config.pagination_style == "offset":
            lines.append(
                f"{self._indent}offset: int = Query(default=0, ge=0),"
            )
            lines.append(
                f"{self._indent}limit: int = Query("
                f"default={self._config.default_page_size}, "
                f"ge=1, le={self._config.max_page_size}),"
            )
        elif self._config.pagination_style == "page_number":
            lines.append(
                f"{self._indent}page: int = Query(default=1, ge=1),"
            )
            lines.append(
                f"{self._indent}page_size: int = Query("
                f"default={self._config.default_page_size}, "
                f"ge=1, le={self._config.max_page_size}),"
            )
        elif self._config.pagination_style == "cursor":
            lines.append(
                f"{self._indent}cursor: Optional[str] = Query(default=None),"
            )
            lines.append(
                f"{self._indent}limit: int = Query("
                f"default={self._config.default_page_size}, "
                f"ge=1, le={self._config.max_page_size}),"
            )

        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> {class_name}ListResponse:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""'
                f"List {to_plural(class_name)} with pagination."
                f'"""'
            )

        # Count query
        lines.append(
            f"{self._indent}count_stmt = select(func.count()).select_from("
            f"{class_name})"
        )
        lines.append(
            f"{self._indent}total_result = await db.execute(count_stmt)"
        )
        lines.append(
            f"{self._indent}total = total_result.scalar_one()"
        )

        # Data query
        lines.append(
            f"{self._indent}stmt = select({class_name})"
        )

        if self._config.pagination_style == "offset":
            lines.append(
                f"{self._indent}stmt = stmt.offset(offset).limit(limit)"
            )
        elif self._config.pagination_style == "page_number":
            lines.append(
                f"{self._indent}computed_offset = (page - 1) * page_size"
            )
            lines.append(
                f"{self._indent}stmt = stmt.offset(computed_offset)"
                f".limit(page_size)"
            )
        elif self._config.pagination_style == "cursor":
            pk_field: str = column_to_field_name(pk_col if pk_col else "id")
            lines.append(
                f"{self._indent}if cursor is not None:"
            )
            lines.append(
                f"{self._double_indent}stmt = stmt.where("
                f"{class_name}.{pk_field} > cursor)"
            )
            lines.append(
                f"{self._indent}stmt = stmt.limit(limit)"
            )

        lines.append(
            f"{self._indent}result = await db.execute(stmt)"
        )
        lines.append(
            f"{self._indent}rows = result.scalars().all()"
        )
        lines.append(
            f"{self._indent}items = ["
            f"{class_name}Read.model_validate(r) for r in rows]"
        )

        # Build response
        if self._config.pagination_style == "offset":
            lines.append(
                f"{self._indent}return {class_name}ListResponse("
            )
            lines.append(f"{self._double_indent}items=items,")
            lines.append(f"{self._double_indent}total=total,")
            lines.append(f"{self._double_indent}offset=offset,")
            lines.append(f"{self._double_indent}limit=limit,")
            lines.append(
                f"{self._double_indent}has_more=(offset + limit) < total,"
            )
            lines.append(f"{self._indent})")
        elif self._config.pagination_style == "page_number":
            lines.append(
                f"{self._indent}total_pages = (total + page_size - 1)"
                f" // page_size"
            )
            lines.append(
                f"{self._indent}return {class_name}ListResponse("
            )
            lines.append(f"{self._double_indent}items=items,")
            lines.append(f"{self._double_indent}total=total,")
            lines.append(f"{self._double_indent}page=page,")
            lines.append(f"{self._double_indent}page_size=page_size,")
            lines.append(
                f"{self._double_indent}total_pages=total_pages,"
            )
            lines.append(
                f"{self._double_indent}has_more=page < total_pages,"
            )
            lines.append(f"{self._indent})")
        elif self._config.pagination_style == "cursor":
            lines.append(
                f"{self._indent}next_cursor_val = ("
                f"str(getattr(rows[-1], '{pk_field}')) "
                f"if rows else None)"
            )
            lines.append(
                f"{self._indent}return {class_name}ListResponse("
            )
            lines.append(f"{self._double_indent}items=items,")
            lines.append(f"{self._double_indent}total=total,")
            lines.append(
                f"{self._double_indent}next_cursor=next_cursor_val,"
            )
            lines.append(
                f"{self._double_indent}has_more=len(rows) == limit,"
            )
            lines.append(f"{self._indent})")

        return lines

    def _gen_update_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
        pk_col: str,
        pk_type: str,
    ) -> List[str]:
        """Generate PATCH /{id} update endpoint."""
        pk_field: str = column_to_field_name(pk_col)
        lines: List[str] = []

        lines.append(f"@router.patch(")
        lines.append(f'{self._indent}"/{{{pk_field}}}",')
        lines.append(f"{self._indent}response_model={class_name}Read,")
        lines.append(
            f'{self._indent}summary="Update a {class_name}",'
        )
        lines.append(f")")
        lines.append(f"async def update_{snake_name}(")
        lines.append(
            f"{self._indent}{pk_field}: {pk_type} = Path(...),"
        )
        lines.append(
            f"{self._indent}payload: {class_name}Update = ..., "
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> {class_name}Read:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""'
                f"Update a {class_name} by primary key (partial update)."
                f'"""'
            )

        # Fetch existing
        lines.append(
            f"{self._indent}stmt = select({class_name}).where("
            f"{class_name}.{pk_field} == {pk_field})"
        )
        lines.append(
            f"{self._indent}result = await db.execute(stmt)"
        )
        lines.append(
            f"{self._indent}db_obj = result.scalar_one_or_none()"
        )
        lines.append(f"{self._indent}if db_obj is None:")
        lines.append(
            f"{self._double_indent}raise HTTPException("
            f"status_code=404, "
            f'detail="{class_name} not found.")'
        )

        # Apply updates
        lines.append(
            f"{self._indent}update_data = payload.model_dump("
            f"exclude_unset=True)"
        )
        lines.append(
            f"{self._indent}for field, value in update_data.items():"
        )
        lines.append(
            f"{self._double_indent}setattr(db_obj, field, value)"
        )
        lines.append(f"{self._indent}await db.commit()")
        lines.append(f"{self._indent}await db.refresh(db_obj)")
        lines.append(
            f"{self._indent}return {class_name}Read.model_validate(db_obj)"
        )
        return lines

    def _gen_delete_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
        pk_col: str,
        pk_type: str,
    ) -> List[str]:
        """Generate DELETE /{id} endpoint."""
        pk_field: str = column_to_field_name(pk_col)
        lines: List[str] = []

        lines.append(f"@router.delete(")
        lines.append(f'{self._indent}"/{{{pk_field}}}",')
        lines.append(f"{self._indent}status_code=204,")
        lines.append(
            f'{self._indent}summary="Delete a {class_name}",'
        )
        lines.append(f")")
        lines.append(f"async def delete_{snake_name}(")
        lines.append(
            f"{self._indent}{pk_field}: {pk_type} = Path(...),"
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> None:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Delete a {class_name} by primary key."""'
            )

        lines.append(
            f"{self._indent}stmt = select({class_name}).where("
            f"{class_name}.{pk_field} == {pk_field})"
        )
        lines.append(
            f"{self._indent}result = await db.execute(stmt)"
        )
        lines.append(
            f"{self._indent}db_obj = result.scalar_one_or_none()"
        )
        lines.append(f"{self._indent}if db_obj is None:")
        lines.append(
            f"{self._double_indent}raise HTTPException("
            f"status_code=404, "
            f'detail="{class_name} not found.")'
        )
        lines.append(f"{self._indent}await db.delete(db_obj)")
        lines.append(f"{self._indent}await db.commit()")
        return lines

    def _gen_search_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
    ) -> List[str]:
        """Generate GET /search endpoint with filterable string columns."""
        lines: List[str] = []
        plural_name: str = to_plural(snake_name)

        # Identify searchable columns (strings only)
        searchable: List[ColumnInfo] = [
            c for c in table.columns
            if c.is_string and not c.primary_key
        ]

        lines.append(f"@router.get(")
        lines.append(f'{self._indent}"/search",')
        lines.append(
            f"{self._indent}response_model={class_name}ListResponse,"
        )
        lines.append(
            f'{self._indent}summary="Search {to_plural(class_name)}",'
        )
        lines.append(f")")
        lines.append(f"async def search_{plural_name}(")

        if searchable:
            lines.append(
                f"{self._indent}q: Optional[str] = Query("
                f'default=None, description="Search query"),'
            )

        lines.append(
            f"{self._indent}offset: int = Query(default=0, ge=0),"
        )
        lines.append(
            f"{self._indent}limit: int = Query("
            f"default={self._config.default_page_size}, "
            f"ge=1, le={self._config.max_page_size}),"
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> {class_name}ListResponse:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""'
                f"Search {to_plural(class_name)} by text fields."
                f'"""'
            )

        lines.append(
            f"{self._indent}stmt = select({class_name})"
        )

        if searchable:
            lines.append(f"{self._indent}if q is not None:")
            lines.append(
                f"{self._double_indent}search_filter = q.strip()"
            )
            lines.append(
                f"{self._double_indent}if search_filter:"
            )

            # Build OR filter across all searchable columns
            if len(searchable) == 1:
                col_field: str = column_to_field_name(searchable[0].name)
                lines.append(
                    f"{self._triple_indent}stmt = stmt.where("
                    f"{class_name}.{col_field}.ilike("
                    f'f"%{{search_filter}}%"))'
                )
            else:
                lines.append(
                    f"{self._triple_indent}from sqlalchemy import or_"
                )
                lines.append(
                    f"{self._triple_indent}stmt = stmt.where(or_("
                )
                for i, sc in enumerate(searchable):
                    col_field = column_to_field_name(sc.name)
                    comma: str = "," if i < len(searchable) - 1 else ""
                    lines.append(
                        f"{self._triple_indent}    "
                        f"{class_name}.{col_field}.ilike("
                        f'f"%{{search_filter}}%"){comma}'
                    )
                lines.append(f"{self._triple_indent}))")

        # Count
        lines.append(
            f"{self._indent}count_stmt = select("
            f"func.count()).select_from(stmt.subquery())"
        )
        lines.append(
            f"{self._indent}total_result = await db.execute(count_stmt)"
        )
        lines.append(
            f"{self._indent}total = total_result.scalar_one()"
        )

        lines.append(
            f"{self._indent}stmt = stmt.offset(offset).limit(limit)"
        )
        lines.append(
            f"{self._indent}result = await db.execute(stmt)"
        )
        lines.append(
            f"{self._indent}rows = result.scalars().all()"
        )
        lines.append(
            f"{self._indent}items = ["
            f"{class_name}Read.model_validate(r) for r in rows]"
        )
        lines.append(
            f"{self._indent}return {class_name}ListResponse("
        )
        lines.append(f"{self._double_indent}items=items,")
        lines.append(f"{self._double_indent}total=total,")
        lines.append(f"{self._double_indent}offset=offset,")
        lines.append(f"{self._double_indent}limit=limit,")
        lines.append(
            f"{self._double_indent}has_more=(offset + limit) < total,"
        )
        lines.append(f"{self._indent})")
        return lines

    def _gen_bulk_create_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
    ) -> List[str]:
        """Generate POST /bulk endpoint for batch creation."""
        plural_name: str = to_plural(snake_name)
        lines: List[str] = []

        lines.append(f"@router.post(")
        lines.append(f'{self._indent}"/bulk",')
        lines.append(
            f"{self._indent}response_model=List[{class_name}Read],"
        )
        lines.append(f"{self._indent}status_code=201,")
        lines.append(
            f'{self._indent}summary="Bulk create {to_plural(class_name)}",'
        )
        lines.append(f")")
        lines.append(f"async def bulk_create_{plural_name}(")
        lines.append(
            f"{self._indent}payloads: List[{class_name}Create],"
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> List[{class_name}Read]:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Create multiple {to_plural(class_name)} '
                f'in a single transaction."""'
            )

        lines.append(
            f"{self._indent}db_objects = ["
        )
        lines.append(
            f"{self._double_indent}{class_name}("
            f"**p.model_dump(exclude_unset=True))"
        )
        lines.append(
            f"{self._double_indent}for p in payloads"
        )
        lines.append(f"{self._indent}]")
        lines.append(f"{self._indent}db.add_all(db_objects)")
        lines.append(f"{self._indent}await db.commit()")
        lines.append(
            f"{self._indent}for obj in db_objects:"
        )
        lines.append(
            f"{self._double_indent}await db.refresh(obj)"
        )
        lines.append(
            f"{self._indent}return ["
            f"{class_name}Read.model_validate(o) for o in db_objects]"
        )
        return lines

    def _gen_bulk_delete_endpoint(
        self,
        table: TableInfo,
        class_name: str,
        snake_name: str,
        pk_col: str,
        pk_type: str,
    ) -> List[str]:
        """Generate DELETE /bulk endpoint for batch deletion."""
        pk_field: str = column_to_field_name(pk_col)
        plural_name: str = to_plural(snake_name)
        lines: List[str] = []

        lines.append(f"@router.delete(")
        lines.append(f'{self._indent}"/bulk",')
        lines.append(f"{self._indent}status_code=204,")
        lines.append(
            f'{self._indent}summary="Bulk delete {to_plural(class_name)}",'
        )
        lines.append(f")")
        lines.append(f"async def bulk_delete_{plural_name}(")
        lines.append(
            f"{self._indent}ids: List[{pk_type}],"
        )
        lines.append(
            f"{self._indent}db: AsyncSession = Depends(get_db),"
        )
        lines.append(f") -> None:")
        if self._config.generate_docstrings:
            lines.append(
                f'{self._indent}"""Delete multiple '
                f'{to_plural(class_name)} by IDs."""'
            )

        lines.append(
            f"{self._indent}stmt = delete({class_name}).where("
            f"{class_name}.{pk_field}.in_(ids))"
        )
        lines.append(
            f"{self._indent}await db.execute(stmt)"
        )
        lines.append(f"{self._indent}await db.commit()")
        return lines

    # ===================================================================
    # 4. Database configuration (engine + session)
    # ===================================================================

    def generate_database_config(self) -> str:
        """Generate database.py with engine, session, and Base."""
        lines: List[str] = []

        lines.append(f'"""')
        lines.append(f"Database engine, session, and declarative base.")
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")

        if self._config.use_async:
            lines.append("from collections.abc import AsyncGenerator")
            lines.append("")
            lines.append(
                "from sqlalchemy.ext.asyncio import ("
            )
            lines.append(f"{self._indent}AsyncSession,")
            lines.append(f"{self._indent}async_sessionmaker,")
            lines.append(f"{self._indent}create_async_engine,")
            lines.append(")")
        else:
            lines.append("from sqlalchemy import create_engine")
            lines.append(
                "from sqlalchemy.orm import Session, sessionmaker"
            )

        lines.append(
            "from sqlalchemy.orm import DeclarativeBase"
        )
        lines.append("")
        lines.append("")

        # Database URL
        lines.append(
            f'DATABASE_URL: str = "{self._config.database_url}"'
        )
        lines.append("")
        lines.append("")

        # Base class
        lines.append("class Base(DeclarativeBase):")
        lines.append(f'{self._indent}"""SQLAlchemy declarative base."""')
        lines.append(f"{self._indent}pass")
        lines.append("")
        lines.append("")

        # Engine
        if self._config.use_async:
            lines.append("engine = create_async_engine(")
            lines.append(f"{self._indent}DATABASE_URL,")
            lines.append(f"{self._indent}echo=False,")
            lines.append(f"{self._indent}pool_pre_ping=True,")
            lines.append(f"{self._indent}pool_size=20,")
            lines.append(f"{self._indent}max_overflow=10,")
            lines.append(")")
            lines.append("")

            # Session factory
            lines.append("async_session_factory = async_sessionmaker(")
            lines.append(f"{self._indent}engine,")
            lines.append(f"{self._indent}class_=AsyncSession,")
            lines.append(f"{self._indent}expire_on_commit=False,")
            lines.append(")")
            lines.append("")
            lines.append("")

            # Dependency
            lines.append(
                "async def get_db() -> AsyncGenerator[AsyncSession, None]:"
            )
            lines.append(
                f'{self._indent}"""FastAPI dependency — yields an async DB session."""'
            )
            lines.append(
                f"{self._indent}async with async_session_factory() as session:"
            )
            lines.append(
                f"{self._double_indent}try:"
            )
            lines.append(
                f"{self._triple_indent}yield session"
            )
            lines.append(
                f"{self._double_indent}finally:"
            )
            lines.append(
                f"{self._triple_indent}await session.close()"
            )
        else:
            lines.append("engine = create_engine(")
            lines.append(f"{self._indent}DATABASE_URL,")
            lines.append(f"{self._indent}echo=False,")
            lines.append(f"{self._indent}pool_pre_ping=True,")
            lines.append(f"{self._indent}pool_size=20,")
            lines.append(f"{self._indent}max_overflow=10,")
            lines.append(")")
            lines.append("")

            lines.append("SessionLocal = sessionmaker(")
            lines.append(f"{self._indent}bind=engine,")
            lines.append(f"{self._indent}autocommit=False,")
            lines.append(f"{self._indent}autoflush=False,")
            lines.append(")")
            lines.append("")
            lines.append("")

            lines.append(
                "def get_db() -> Session:"
            )
            lines.append(
                f'{self._indent}"""FastAPI dependency — yields a DB session."""'
            )
            lines.append(
                f"{self._indent}db = SessionLocal()"
            )
            lines.append(f"{self._indent}try:")
            lines.append(f"{self._double_indent}yield db")
            lines.append(f"{self._indent}finally:")
            lines.append(f"{self._double_indent}db.close()")

        lines.append("")

        content: str = "\n".join(lines)
        logger.debug(
            "Generated database config: %d lines.",
            content.count("\n") + 1,
        )
        return content

    # ===================================================================
    # 5. Application entry point (main.py)
    # ===================================================================

    def generate_main_app(
        self, schema: SchemaDefinition
    ) -> str:
        """Generate the FastAPI application entry point (main.py)."""
        lines: List[str] = []

        lines.append(f'"""')
        lines.append(
            f"{self._config.project_name} — FastAPI Application"
        )
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append("from contextlib import asynccontextmanager")
        lines.append("from collections.abc import AsyncGenerator")
        lines.append("")
        lines.append("from fastapi import FastAPI")

        if self._config.enable_cors:
            lines.append(
                "from fastapi.middleware.cors import CORSMiddleware"
            )

        lines.append("")
        lines.append("from apigen_generated.database import engine, Base")
        lines.append("")

        # Import routers
        for table in schema.tables:
            snake: str = to_snake_case(table.name)
            lines.append(
                f"from apigen_generated.routers.{snake} import "
                f"router as {snake}_router"
            )

        lines.append("")
        lines.append("")

        # Lifespan
        lines.append("@asynccontextmanager")
        lines.append(
            "async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:"
        )
        lines.append(
            f'{self._indent}"""Application lifespan — create tables on startup."""'
        )
        if self._config.use_async:
            lines.append(
                f"{self._indent}async with engine.begin() as conn:"
            )
            lines.append(
                f"{self._double_indent}await conn.run_sync("
                f"Base.metadata.create_all)"
            )
        else:
            lines.append(
                f"{self._indent}Base.metadata.create_all(bind=engine)"
            )
        lines.append(f"{self._indent}yield")
        if self._config.use_async:
            lines.append(f"{self._indent}await engine.dispose()")

        lines.append("")
        lines.append("")

        # App instance
        lines.append("app = FastAPI(")
        lines.append(
            f'{self._indent}title="{self._config.project_name}",'
        )
        lines.append(
            f'{self._indent}version="{self._config.project_version}",'
        )
        lines.append(
            f'{self._indent}description="{self._config.project_description}",'
        )
        lines.append(f"{self._indent}lifespan=lifespan,")
        lines.append(")")
        lines.append("")

        # CORS
        if self._config.enable_cors:
            origins_str: str = ", ".join(
                f'"{o}"' for o in self._config.cors_origins
            )
            lines.append("app.add_middleware(")
            lines.append(f"{self._indent}CORSMiddleware,")
            lines.append(
                f"{self._indent}allow_origins=[{origins_str}],"
            )
            lines.append(
                f"{self._indent}allow_credentials=True,"
            )
            lines.append(
                f'{self._indent}allow_methods=["*"],'
            )
            lines.append(
                f'{self._indent}allow_headers=["*"],'
            )
            lines.append(")")
            lines.append("")

        # Include routers
        for table in schema.tables:
            snake = to_snake_case(table.name)
            lines.append(
                f"app.include_router("
                f"{snake}_router, "
                f'prefix="{self._config.api_prefix}")'
            )

        lines.append("")
        lines.append("")

        # Health check
        lines.append('@app.get("/health", tags=["Health"])')
        lines.append("async def health_check() -> Dict[str, str]:")
        lines.append(
            f'{self._indent}"""Health check endpoint."""'
        )
        lines.append(
            f'{self._indent}return {{"status": "healthy", '
            f'"version": "{self._config.project_version}"}}'
        )
        lines.append("")

        # Add import for Dict
        content: str = "\n".join(lines)
        # Inject typing import at the top
        content = content.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n\n"
            "from typing import Dict\n",
            1,
        )

        logger.debug(
            "Generated main.py: %d lines.",
            content.count("\n") + 1,
        )
        return content

    # ===================================================================
    # 6. Alembic environment
    # ===================================================================

    def generate_alembic_env(self) -> str:
        """Generate Alembic env.py for migration management."""
        lines: List[str] = []

        lines.append(f'"""')
        lines.append(f"Alembic environment configuration.")
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append("import asyncio")
        lines.append("from logging.config import fileConfig")
        lines.append("")
        lines.append("from alembic import context")
        lines.append("from sqlalchemy import pool")

        if self._config.use_async:
            lines.append(
                "from sqlalchemy.ext.asyncio import "
                "async_engine_from_config"
            )
        else:
            lines.append(
                "from sqlalchemy import engine_from_config"
            )

        lines.append("")
        lines.append(
            "from apigen_generated.database import Base, DATABASE_URL"
        )
        lines.append("")

        # Alembic Config
        lines.append("config = context.config")
        lines.append("")
        lines.append("if config.config_file_name is not None:")
        lines.append(f"{self._indent}fileConfig(config.config_file_name)")
        lines.append("")
        lines.append(
            'config.set_main_option("sqlalchemy.url", DATABASE_URL)'
        )
        lines.append("")
        lines.append("target_metadata = Base.metadata")
        lines.append("")
        lines.append("")

        # Offline mode
        lines.append("def run_migrations_offline() -> None:")
        lines.append(
            f'{self._indent}"""Run migrations in offline mode."""'
        )
        lines.append(
            f'{self._indent}url = config.get_main_option("sqlalchemy.url")'
        )
        lines.append(f"{self._indent}context.configure(")
        lines.append(f"{self._double_indent}url=url,")
        lines.append(
            f"{self._double_indent}target_metadata=target_metadata,"
        )
        lines.append(f"{self._double_indent}literal_binds=True,")
        lines.append(
            f'{self._double_indent}dialect_opts={{"paramstyle": "named"}},'
        )
        lines.append(f"{self._indent})")
        lines.append(
            f"{self._indent}with context.begin_transaction():"
        )
        lines.append(
            f"{self._double_indent}context.run_migrations()"
        )
        lines.append("")
        lines.append("")

        # Online mode
        if self._config.use_async:
            lines.append(
                "def do_run_migrations(connection) -> None:"
            )
            lines.append(f"{self._indent}context.configure(")
            lines.append(
                f"{self._double_indent}connection=connection,"
            )
            lines.append(
                f"{self._double_indent}target_metadata=target_metadata,"
            )
            lines.append(f"{self._indent})")
            lines.append(
                f"{self._indent}with context.begin_transaction():"
            )
            lines.append(
                f"{self._double_indent}context.run_migrations()"
            )
            lines.append("")
            lines.append("")

            lines.append(
                "async def run_async_migrations() -> None:"
            )
            lines.append(
                f'{self._indent}"""Run migrations in online (async) mode."""'
            )
            lines.append(
                f"{self._indent}connectable = "
                f"async_engine_from_config("
            )
            lines.append(
                f"{self._double_indent}"
                f"config.get_section(config.config_ini_section, {{}}),"
            )
            lines.append(
                f"{self._double_indent}prefix=\"sqlalchemy.\","
            )
            lines.append(
                f"{self._double_indent}poolclass=pool.NullPool,"
            )
            lines.append(f"{self._indent})")
            lines.append("")
            lines.append(
                f"{self._indent}async with connectable.connect() as connection:"
            )
            lines.append(
                f"{self._double_indent}await connection.run_sync("
                f"do_run_migrations)"
            )
            lines.append("")
            lines.append(
                f"{self._indent}await connectable.dispose()"
            )
            lines.append("")
            lines.append("")

            lines.append("def run_migrations_online() -> None:")
            lines.append(
                f'{self._indent}"""Run migrations in online mode."""'
            )
            lines.append(
                f"{self._indent}asyncio.run(run_async_migrations())"
            )
        else:
            lines.append("def run_migrations_online() -> None:")
            lines.append(
                f'{self._indent}"""Run migrations in online mode."""'
            )
            lines.append(
                f"{self._indent}connectable = engine_from_config("
            )
            lines.append(
                f"{self._double_indent}"
                f"config.get_section(config.config_ini_section, {{}}),"
            )
            lines.append(
                f"{self._double_indent}prefix=\"sqlalchemy.\","
            )
            lines.append(
                f"{self._double_indent}poolclass=pool.NullPool,"
            )
            lines.append(f"{self._indent})")
            lines.append("")
            lines.append(
                f"{self._indent}with connectable.connect() as connection:"
            )
            lines.append(f"{self._double_indent}context.configure(")
            lines.append(
                f"{self._triple_indent}connection=connection,"
            )
            lines.append(
                f"{self._triple_indent}target_metadata=target_metadata,"
            )
            lines.append(f"{self._double_indent})")
            lines.append(
                f"{self._double_indent}with context.begin_transaction():"
            )
            lines.append(
                f"{self._triple_indent}context.run_migrations()"
            )

        lines.append("")
        lines.append("")

        # Entrypoint
        lines.append("if context.is_offline_mode():")
        lines.append(f"{self._indent}run_migrations_offline()")
        lines.append("else:")
        lines.append(f"{self._indent}run_migrations_online()")
        lines.append("")

        content: str = "\n".join(lines)
        logger.debug(
            "Generated alembic env.py: %d lines.",
            content.count("\n") + 1,
        )
        return content

    # ===================================================================
    # 7. __init__.py files
    # ===================================================================

    def generate_init_file(
        self,
        module_name: str,
        imports: Optional[List[str]] = None,
    ) -> str:
        """Generate an __init__.py file with optional re-exports."""
        lines: List[str] = []
        lines.append(f'"""')
        lines.append(f"{module_name} package.")
        lines.append(f"Auto-generated by NexaFlow APIGen.")
        lines.append(f'"""')
        lines.append("")

        if imports:
            for imp in imports:
                lines.append(imp)
            lines.append("")

        content: str = "\n".join(lines)
        return content

    # ===================================================================
    # 8. Aggregate generation (all files for one table)
    # ===================================================================

    def generate_all_for_table(
        self, table: TableInfo
    ) -> Dict[str, str]:
        """
        Generate all code files for a single table.

        Returns a dict of relative_path → file_content.

        Complexity: O(C + R + I) where C = columns, R = relationships,
        I = indexes for this table.
        """
        snake_name: str = to_snake_case(table.name)
        result: Dict[str, str] = {}

        result[f"models/{snake_name}.py"] = self.generate_orm_model(table)
        result[f"schemas/{snake_name}.py"] = self.generate_pydantic_schemas(table)
        result[f"routers/{snake_name}.py"] = self.generate_router(table)

        logger.debug(
            "Generated all files for table '%s': %d files.",
            table.name,
            len(result),
        )
        return result

    def generate_all(
        self, schema: SchemaDefinition
    ) -> Dict[str, str]:
        """
        Generate the complete project codebase.

        Returns a dict of relative_path → file_content.

        Complexity: O(T × (C + R + I)) — linear in total schema entities.
        """
        result: Dict[str, str] = {}

        # Database config
        result["database.py"] = self.generate_database_config()

        # Main app
        result["main.py"] = self.generate_main_app(schema)

        # Per-table files
        model_imports: List[str] = []
        schema_imports: List[str] = []
        router_imports: List[str] = []

        for table in schema.tables:
            table_files: Dict[str, str] = self.generate_all_for_table(table)
            result.update(table_files)

            snake: str = to_snake_case(table.name)
            cls: str = table_to_class_name(table.name)
            model_imports.append(
                f"from .{snake} import {cls}"
            )
            schema_imports.append(
                f"from .{snake} import ("
                f"{cls}Base, {cls}Create, {cls}Read, "
                f"{cls}Update, {cls}ListResponse)"
            )
            router_imports.append(
                f"from .{snake} import router as {snake}_router"
            )

        # Package __init__.py files
        result["__init__.py"] = self.generate_init_file(
            self._config.project_name
        )
        result["models/__init__.py"] = self.generate_init_file(
            "models", model_imports
        )
        result["schemas/__init__.py"] = self.generate_init_file(
            "schemas", schema_imports
        )
        result["routers/__init__.py"] = self.generate_init_file(
            "routers", router_imports
        )

        # Alembic
        if self._config.generate_alembic:
            result["alembic/env.py"] = self.generate_alembic_env()
            result["alembic/__init__.py"] = self.generate_init_file("alembic")

        total_lines: int = sum(
            content.count("\n") + 1 for content in result.values()
        )
        logger.info(
            "Full generation complete: %d files, ~%d lines.",
            len(result),
            total_lines,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "TemplateGenerator",
]

logger.debug("apigen.templates loaded.")
