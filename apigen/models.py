# File: apigen/models.py
"""
NexaFlow APIGen - Core Data Models
===================================
Pydantic V2 models representing database schema elements and code generation
configuration. These models form the single source of truth for the entire
pipeline: Schema Parsing → Validation → Code Generation → Export.

Performance target: All model instantiation and serialization must operate
at O(1) per-entity to support 50,000+ line generation in <30 seconds.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen.models")

# ---------------------------------------------------------------------------
# Enums — fixed sets used across the entire project
# ---------------------------------------------------------------------------


class ColumnType(str, Enum):
    """Supported SQL / SQLAlchemy column types."""

    # Numeric
    INTEGER = "integer"
    BIGINTEGER = "biginteger"
    SMALLINTEGER = "smallinteger"
    FLOAT = "float"
    NUMERIC = "numeric"
    DOUBLE = "double"
    BOOLEAN = "boolean"

    # String / Binary
    STRING = "string"
    TEXT = "text"
    VARCHAR = "varchar"
    CHAR = "char"
    BINARY = "binary"
    LARGEBINARY = "largebinary"

    # Date / Time
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    TIMESTAMP = "timestamp"
    INTERVAL = "interval"

    # Special
    UUID = "uuid"
    JSON = "json"
    JSONB = "jsonb"
    ARRAY = "array"
    ENUM = "enum"
    HSTORE = "hstore"
    INET = "inet"
    CIDR = "cidr"
    MACADDR = "macaddr"


class RelationshipType(str, Enum):
    """ORM relationship cardinalities."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class IndexType(str, Enum):
    """Supported index kinds."""

    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"
    UNIQUE = "unique"


class OnDeleteAction(str, Enum):
    """Foreign-key ON DELETE behaviour."""

    CASCADE = "CASCADE"
    SET_NULL = "SET NULL"
    SET_DEFAULT = "SET DEFAULT"
    RESTRICT = "RESTRICT"
    NO_ACTION = "NO ACTION"


class OnUpdateAction(str, Enum):
    """Foreign-key ON UPDATE behaviour."""

    CASCADE = "CASCADE"
    SET_NULL = "SET NULL"
    SET_DEFAULT = "SET DEFAULT"
    RESTRICT = "RESTRICT"
    NO_ACTION = "NO ACTION"


class AuthStrategy(str, Enum):
    """Supported authentication strategies for generated APIs."""

    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"


class PaginationStyle(str, Enum):
    """Pagination flavours for list endpoints."""

    OFFSET = "offset"
    CURSOR = "cursor"
    PAGE_NUMBER = "page_number"


class NamingConvention(str, Enum):
    """Code-generation naming conventions."""

    SNAKE_CASE = "snake_case"
    CAMEL_CASE = "camelCase"
    PASCAL_CASE = "PascalCase"


class DatabaseDialect(str, Enum):
    """Target database dialects."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    ORACLE = "oracle"


# ---------------------------------------------------------------------------
# Mixin: shared model configuration
# ---------------------------------------------------------------------------

_SHARED_CONFIG: ConfigDict = ConfigDict(
    strict=False,
    populate_by_name=True,
    validate_assignment=True,
    use_enum_values=True,
    frozen=False,
    extra="forbid",
)


# ---------------------------------------------------------------------------
# Low-level schema primitives
# ---------------------------------------------------------------------------


class ColumnConstraint(BaseModel):
    """A single column-level constraint (CHECK, DEFAULT, etc.)."""

    model_config = _SHARED_CONFIG

    name: Optional[str] = Field(default=None, description="Constraint name (optional).")
    expression: str = Field(
        ...,
        min_length=1,
        description="SQL expression, e.g. 'age > 0' or 'now()'.",
    )
    constraint_type: Literal["check", "default", "unique", "exclude"] = Field(
        ..., description="Kind of constraint."
    )

    def __repr__(self) -> str:
        return f"<ColumnConstraint {self.constraint_type}: {self.expression}>"


class EnumDefinition(BaseModel):
    """Represents an SQL ENUM type."""

    model_config = _SHARED_CONFIG

    name: str = Field(..., min_length=1, description="Enum type name.")
    values: List[str] = Field(
        ..., min_length=1, description="Allowed values for this enum."
    )
    schema_name: Optional[str] = Field(
        default=None, description="Database schema (e.g. 'public')."
    )

    @field_validator("values")
    @classmethod
    def _unique_values(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            dupes: List[str] = [x for x in v if v.count(x) > 1]
            raise ValueError(f"Duplicate enum values detected: {dupes}")
        return v


class ColumnDefault(BaseModel):
    """Column default value descriptor."""

    model_config = _SHARED_CONFIG

    is_server_default: bool = Field(
        default=False,
        description="True if the default is evaluated server-side (e.g. now()).",
    )
    value: Any = Field(..., description="Default value or SQL expression string.")
    is_clause: bool = Field(
        default=False,
        description="True when value is raw SQL text, not a literal.",
    )


class ColumnInfo(BaseModel):
    """
    Complete specification of a single database column.

    This is the most granular building block.  Every column in every table
    in the parsed schema becomes exactly one ``ColumnInfo`` instance.
    """

    model_config = _SHARED_CONFIG

    name: str = Field(..., min_length=1, description="Column name.")
    column_type: ColumnType = Field(..., description="Abstract SQL type.")
    max_length: Optional[int] = Field(
        default=None, ge=1, description="Max length for VARCHAR / CHAR."
    )
    precision: Optional[int] = Field(
        default=None, ge=0, description="Precision for NUMERIC / DECIMAL."
    )
    scale: Optional[int] = Field(
        default=None, ge=0, description="Scale for NUMERIC / DECIMAL."
    )
    nullable: bool = Field(default=True, description="Whether the column allows NULL.")
    primary_key: bool = Field(default=False, description="Part of the primary key?")
    autoincrement: bool = Field(
        default=False, description="Auto-increment (SERIAL, IDENTITY, etc.)."
    )
    unique: bool = Field(default=False, description="Has a UNIQUE constraint?")
    index: bool = Field(default=False, description="Should a single-column index be created?")
    default: Optional[ColumnDefault] = Field(
        default=None, description="Default value descriptor."
    )
    server_default: Optional[str] = Field(
        default=None, description="Raw SQL server_default expression."
    )
    comment: Optional[str] = Field(default=None, description="Column comment / doc.")
    constraints: List[ColumnConstraint] = Field(
        default_factory=list, description="Extra constraints."
    )
    enum_definition: Optional[EnumDefinition] = Field(
        default=None,
        description="Enum type info (only when column_type == 'enum').",
    )
    array_item_type: Optional[ColumnType] = Field(
        default=None,
        description="Element type when column_type == 'array'.",
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON Schema for json/jsonb columns.",
    )

    # -- Derived helpers (computed_field keeps them out of __init__) ---------

    @computed_field  # type: ignore[misc]
    @property
    def is_numeric(self) -> bool:
        return self.column_type in {
            ColumnType.INTEGER,
            ColumnType.BIGINTEGER,
            ColumnType.SMALLINTEGER,
            ColumnType.FLOAT,
            ColumnType.NUMERIC,
            ColumnType.DOUBLE,
        }

    @computed_field  # type: ignore[misc]
    @property
    def is_string(self) -> bool:
        return self.column_type in {
            ColumnType.STRING,
            ColumnType.TEXT,
            ColumnType.VARCHAR,
            ColumnType.CHAR,
        }

    @computed_field  # type: ignore[misc]
    @property
    def is_temporal(self) -> bool:
        return self.column_type in {
            ColumnType.DATE,
            ColumnType.DATETIME,
            ColumnType.TIME,
            ColumnType.TIMESTAMP,
            ColumnType.INTERVAL,
        }

    @computed_field  # type: ignore[misc]
    @property
    def python_type_hint(self) -> str:
        """Return the corresponding Python type-hint string for code generation."""
        _MAP: Dict[str, str] = {
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
            "json": "Any",
            "jsonb": "Any",
            "array": "List[Any]",
            "enum": "str",
            "hstore": "Dict[str, str]",
            "inet": "str",
            "cidr": "str",
            "macaddr": "str",
        }
        base: str = _MAP.get(self.column_type, "Any")
        if self.nullable and not self.primary_key:
            return f"Optional[{base}]"
        return base

    @computed_field  # type: ignore[misc]
    @property
    def sqlalchemy_type_str(self) -> str:
        """SQLAlchemy type constructor string for code generation."""
        _MAP: Dict[str, str] = {
            "integer": "Integer",
            "biginteger": "BigInteger",
            "smallinteger": "SmallInteger",
            "float": "Float",
            "numeric": f"Numeric(precision={self.precision}, scale={self.scale})"
            if self.precision is not None
            else "Numeric",
            "double": "Double",
            "boolean": "Boolean",
            "string": f"String({self.max_length})" if self.max_length else "String",
            "text": "Text",
            "varchar": f"String({self.max_length})" if self.max_length else "String",
            "char": f"CHAR({self.max_length})" if self.max_length else "CHAR",
            "binary": "LargeBinary",
            "largebinary": "LargeBinary",
            "date": "Date",
            "datetime": "DateTime",
            "time": "Time",
            "timestamp": "DateTime(timezone=True)",
            "interval": "Interval",
            "uuid": "UUID(as_uuid=True)",
            "json": "JSON",
            "jsonb": "JSONB",
            "array": f"ARRAY({self.array_item_type.value.capitalize() if self.array_item_type else 'Text'})",
            "enum": f"SQLEnum({', '.join(repr(v) for v in (self.enum_definition.values if self.enum_definition else []))},"
            f" name='{self.enum_definition.name if self.enum_definition else 'unknown'}')",
            "hstore": "HSTORE",
            "inet": "INET",
            "cidr": "CIDR",
            "macaddr": "MACADDR",
        }
        return _MAP.get(self.column_type, "String")

    # -- Validators ---------------------------------------------------------

    @model_validator(mode="after")
    def _validate_enum_has_definition(self) -> "ColumnInfo":
        if self.column_type == ColumnType.ENUM and self.enum_definition is None:
            raise ValueError(
                f"Column '{self.name}' is of type ENUM but 'enum_definition' is missing."
            )
        return self

    @model_validator(mode="after")
    def _validate_array_item_type(self) -> "ColumnInfo":
        if self.column_type == ColumnType.ARRAY and self.array_item_type is None:
            logger.warning(
                "Column '%s' is ARRAY without explicit array_item_type; "
                "defaulting to TEXT.",
                self.name,
            )
        return self

    def __repr__(self) -> str:
        pk_flag: str = " PK" if self.primary_key else ""
        null_flag: str = " NULL" if self.nullable else " NOT NULL"
        return f"<Column {self.name} {self.column_type.value}{pk_flag}{null_flag}>"


# ---------------------------------------------------------------------------
# Foreign key & relationships
# ---------------------------------------------------------------------------


class ForeignKeyInfo(BaseModel):
    """Describes a single foreign-key reference between two columns."""

    model_config = _SHARED_CONFIG

    constrained_column: str = Field(
        ..., min_length=1, description="Local column name."
    )
    referred_table: str = Field(
        ..., min_length=1, description="Target table name."
    )
    referred_column: str = Field(
        ..., min_length=1, description="Target column name."
    )
    referred_schema: Optional[str] = Field(
        default=None, description="Target schema (if cross-schema)."
    )
    constraint_name: Optional[str] = Field(
        default=None, description="FK constraint name."
    )
    on_delete: OnDeleteAction = Field(
        default=OnDeleteAction.NO_ACTION,
        description="ON DELETE referential action.",
    )
    on_update: OnUpdateAction = Field(
        default=OnUpdateAction.NO_ACTION,
        description="ON UPDATE referential action.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def relationship_attr_name(self) -> str:
        """
        Generate a sensible Python attribute name for the ORM relationship.
        E.g. ``user_id`` → ``user``, ``author_id`` → ``author``.
        """
        col: str = self.constrained_column
        if col.endswith("_id"):
            return col[:-3]
        return f"{self.referred_table}_ref"

    def __repr__(self) -> str:
        return (
            f"<FK {self.constrained_column} → "
            f"{self.referred_table}.{self.referred_column}>"
        )


class RelationshipInfo(BaseModel):
    """Describes an ORM-level relationship (derived from foreign keys)."""

    model_config = _SHARED_CONFIG

    name: str = Field(..., min_length=1, description="Attribute name on the model.")
    relationship_type: RelationshipType = Field(
        ..., description="Cardinality."
    )
    target_table: str = Field(..., min_length=1, description="Related table.")
    foreign_key: ForeignKeyInfo = Field(
        ..., description="The underlying foreign key."
    )
    back_populates: Optional[str] = Field(
        default=None, description="back_populates argument for relationship()."
    )
    lazy: str = Field(
        default="select",
        description="SQLAlchemy lazy loading strategy.",
    )
    uselist: Optional[bool] = Field(
        default=None,
        description="Explicit uselist flag (None = auto-detect from cardinality).",
    )
    secondary_table: Optional[str] = Field(
        default=None,
        description="Association table for M2M relationships.",
    )
    cascade: str = Field(
        default="save-update, merge",
        description="SQLAlchemy cascade string.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def resolved_uselist(self) -> bool:
        if self.uselist is not None:
            return self.uselist
        return self.relationship_type in {
            RelationshipType.ONE_TO_MANY,
            RelationshipType.MANY_TO_MANY,
        }

    def __repr__(self) -> str:
        return f"<Relationship {self.name} ({self.relationship_type}) → {self.target_table}>"


# ---------------------------------------------------------------------------
# Index & Unique Constraint
# ---------------------------------------------------------------------------


class IndexInfo(BaseModel):
    """Composite or single-column index."""

    model_config = _SHARED_CONFIG

    name: str = Field(..., min_length=1, description="Index name.")
    columns: List[str] = Field(
        ..., min_length=1, description="Ordered list of column names."
    )
    unique: bool = Field(default=False, description="UNIQUE index?")
    index_type: IndexType = Field(
        default=IndexType.BTREE, description="Index method."
    )
    condition: Optional[str] = Field(
        default=None, description="Partial index WHERE clause."
    )

    @field_validator("columns")
    @classmethod
    def _no_duplicate_columns(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError(f"Duplicate columns in index: {v}")
        return v


class UniqueConstraintInfo(BaseModel):
    """Multi-column unique constraint."""

    model_config = _SHARED_CONFIG

    name: Optional[str] = Field(default=None, description="Constraint name.")
    columns: List[str] = Field(
        ..., min_length=1, description="Columns in the unique group."
    )


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class TableInfo(BaseModel):
    """
    Complete representation of a single database table / entity.

    This is the central model consumed by the code generator.  One
    ``TableInfo`` instance drives creation of: SQLAlchemy ORM model,
    Pydantic schema(s), CRUD router, Alembic migration stub, and tests.
    """

    model_config = _SHARED_CONFIG

    name: str = Field(..., min_length=1, description="Table name (snake_case).")
    schema_name: Optional[str] = Field(
        default=None, description="Database schema (e.g. 'public')."
    )
    columns: List[ColumnInfo] = Field(
        ..., min_length=1, description="Columns (at least one required)."
    )
    primary_key_columns: List[str] = Field(
        default_factory=list,
        description="Explicit PK column names (auto-detected if empty).",
    )
    foreign_keys: List[ForeignKeyInfo] = Field(
        default_factory=list, description="Foreign key references."
    )
    relationships: List[RelationshipInfo] = Field(
        default_factory=list, description="ORM relationships."
    )
    indexes: List[IndexInfo] = Field(
        default_factory=list, description="Indexes."
    )
    unique_constraints: List[UniqueConstraintInfo] = Field(
        default_factory=list, description="Multi-column unique constraints."
    )
    comment: Optional[str] = Field(default=None, description="Table comment / doc.")
    extend_existing: bool = Field(
        default=True, description="SQLAlchemy __table_args__ extend_existing."
    )

    # -- Computed helpers ---------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def class_name(self) -> str:
        """PascalCase class name derived from table name (O(n) on name length)."""
        return "".join(part.capitalize() for part in self.name.split("_"))

    @computed_field  # type: ignore[misc]
    @property
    def resolved_primary_keys(self) -> List[str]:
        """Return PK column names — explicit list or auto-detect from ColumnInfo."""
        if self.primary_key_columns:
            return self.primary_key_columns
        return [c.name for c in self.columns if c.primary_key]

    @computed_field  # type: ignore[misc]
    @property
    def has_composite_pk(self) -> bool:
        return len(self.resolved_primary_keys) > 1

    @computed_field  # type: ignore[misc]
    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @computed_field  # type: ignore[misc]
    @property
    def nullable_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.nullable and not c.primary_key]

    @computed_field  # type: ignore[misc]
    @property
    def required_columns(self) -> List[str]:
        return [
            c.name
            for c in self.columns
            if (not c.nullable or c.primary_key) and not c.autoincrement
        ]

    @computed_field  # type: ignore[misc]
    @property
    def fk_column_names(self) -> FrozenSet[str]:
        return frozenset(fk.constrained_column for fk in self.foreign_keys)

    @field_serializer("fk_column_names")
    def _serialize_fk_column_names(self, v: FrozenSet[str], _info: Any) -> List[str]:
        return sorted(v)

    # -- Fast O(1) lookup cache (populated once via model_validator) --------
    _column_map: Dict[str, ColumnInfo] = {}

    @model_validator(mode="after")
    def _build_column_map(self) -> "TableInfo":
        object.__setattr__(
            self,
            "_column_map",
            {c.name: c for c in self.columns},
        )
        return self

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        """O(1) column lookup by name."""
        return self._column_map.get(name)

    @model_validator(mode="after")
    def _auto_detect_pks(self) -> "TableInfo":
        if not self.resolved_primary_keys:
            logger.warning(
                "Table '%s' has no primary key columns defined or detected. "
                "Code generation may produce invalid models.",
                self.name,
            )
        return self

    @model_validator(mode="after")
    def _validate_fk_columns_exist(self) -> "TableInfo":
        col_set: Set[str] = {c.name for c in self.columns}
        for fk in self.foreign_keys:
            if fk.constrained_column not in col_set:
                raise ValueError(
                    f"ForeignKey references column '{fk.constrained_column}' "
                    f"which does not exist in table '{self.name}'. "
                    f"Available columns: {sorted(col_set)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_index_columns_exist(self) -> "TableInfo":
        col_set: Set[str] = {c.name for c in self.columns}
        for idx in self.indexes:
            missing: List[str] = [c for c in idx.columns if c not in col_set]
            if missing:
                raise ValueError(
                    f"Index '{idx.name}' on table '{self.name}' references "
                    f"non-existent columns: {missing}"
                )
        return self

    @model_validator(mode="after")
    def _validate_unique_columns_exist(self) -> "TableInfo":
        col_set: Set[str] = {c.name for c in self.columns}
        for uc in self.unique_constraints:
            missing: List[str] = [c for c in uc.columns if c not in col_set]
            if missing:
                raise ValueError(
                    f"UniqueConstraint on table '{self.name}' references "
                    f"non-existent columns: {missing}"
                )
        return self

    def __repr__(self) -> str:
        return (
            f"<Table {self.name} "
            f"({len(self.columns)} cols, {len(self.foreign_keys)} FKs, "
            f"{len(self.relationships)} rels)>"
        )


# ---------------------------------------------------------------------------
# Code Generation Configuration
# ---------------------------------------------------------------------------


class CRUDConfig(BaseModel):
    """Per-table CRUD endpoint toggle."""

    model_config = _SHARED_CONFIG

    create: bool = Field(default=True, description="Generate POST endpoint.")
    read_one: bool = Field(default=True, description="Generate GET /:id endpoint.")
    read_many: bool = Field(default=True, description="Generate GET / (list) endpoint.")
    update: bool = Field(default=True, description="Generate PUT/PATCH endpoint.")
    delete: bool = Field(default=True, description="Generate DELETE endpoint.")
    bulk_create: bool = Field(default=False, description="Generate bulk POST endpoint.")
    bulk_delete: bool = Field(default=False, description="Generate bulk DELETE endpoint.")
    search: bool = Field(default=True, description="Generate search/filter endpoint.")
    export_csv: bool = Field(default=False, description="Generate CSV export endpoint.")


class GenerationConfig(BaseModel):
    """
    Master configuration that controls every aspect of code generation.

    A single instance of this model (combined with a ``SchemaDefinition``)
    is all the generator needs to produce the full output.
    """

    model_config = _SHARED_CONFIG

    # -- Project metadata ---------------------------------------------------
    project_name: str = Field(
        default="myproject",
        min_length=1,
        max_length=128,
        description="Project / package name.",
    )
    project_version: str = Field(
        default="0.1.0", description="Semantic version string."
    )
    project_description: str = Field(
        default="Auto-generated FastAPI application.",
        description="Short description for pyproject.toml / OpenAPI.",
    )
    author: str = Field(default="NexaFlow APIGen", description="Author name.")

    # -- Database -----------------------------------------------------------
    database_dialect: DatabaseDialect = Field(
        default=DatabaseDialect.POSTGRESQL,
        description="Target database backend.",
    )
    database_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/mydb",
        description="SQLAlchemy connection URL (async driver recommended).",
    )
    use_async: bool = Field(
        default=True, description="Generate async SQLAlchemy engine + sessions."
    )

    # -- Code style ---------------------------------------------------------
    naming_convention: NamingConvention = Field(
        default=NamingConvention.SNAKE_CASE,
        description="Naming style for generated identifiers.",
    )
    line_length: int = Field(
        default=99, ge=40, le=200, description="Black / Ruff line length."
    )
    indent_size: int = Field(
        default=4, ge=2, le=8, description="Indentation width."
    )
    generate_docstrings: bool = Field(
        default=True, description="Add docstrings to generated classes."
    )
    generate_type_hints: bool = Field(
        default=True, description="Add type hints everywhere."
    )

    # -- Features -----------------------------------------------------------
    generate_alembic: bool = Field(
        default=True, description="Generate Alembic migration setup."
    )
    generate_tests: bool = Field(
        default=True, description="Generate pytest test stubs."
    )
    generate_docker: bool = Field(
        default=False, description="Generate Dockerfile + docker-compose."
    )
    generate_crud_schemas: bool = Field(
        default=True, description="Generate Pydantic CRUD schemas per table."
    )
    generate_repository_layer: bool = Field(
        default=True,
        description="Generate a repository/DAO layer abstracting raw ORM calls.",
    )

    # -- API ----------------------------------------------------------------
    auth_strategy: AuthStrategy = Field(
        default=AuthStrategy.JWT, description="Authentication strategy."
    )
    pagination_style: PaginationStyle = Field(
        default=PaginationStyle.OFFSET, description="List endpoint pagination."
    )
    default_page_size: int = Field(
        default=25, ge=1, le=1000, description="Default items per page."
    )
    max_page_size: int = Field(
        default=100, ge=1, le=10000, description="Maximum items per page."
    )
    enable_cors: bool = Field(default=True, description="Add CORS middleware.")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins.",
    )
    enable_rate_limiting: bool = Field(
        default=False, description="Add rate-limiting middleware."
    )
    rate_limit_per_minute: int = Field(
        default=60, ge=1, description="Requests per minute when rate limiting is on."
    )
    api_prefix: str = Field(
        default="/api/v1", description="Global API route prefix."
    )
    enable_websockets: bool = Field(
        default=False, description="Generate WebSocket notification endpoints."
    )

    # -- CRUD toggles per table (override defaults) -------------------------
    crud_overrides: Dict[str, CRUDConfig] = Field(
        default_factory=dict,
        description="Per-table CRUD toggles keyed by table name.",
    )
    default_crud: CRUDConfig = Field(
        default_factory=CRUDConfig,
        description="Default CRUD toggle applied when no override exists.",
    )

    # -- Output -------------------------------------------------------------
    output_dir: str = Field(
        default="./generated", description="Root directory for generated code."
    )
    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite files if output_dir already contains code.",
    )

    # -- Helpers ------------------------------------------------------------

    def get_crud_config(self, table_name: str) -> CRUDConfig:
        """O(1) lookup for per-table CRUD config with fallback to default."""
        return self.crud_overrides.get(table_name, self.default_crud)

    @model_validator(mode="after")
    def _validate_page_sizes(self) -> "GenerationConfig":
        if self.default_page_size > self.max_page_size:
            raise ValueError(
                f"default_page_size ({self.default_page_size}) must be "
                f"<= max_page_size ({self.max_page_size})."
            )
        return self


# ---------------------------------------------------------------------------
# Schema Definition — top-level container
# ---------------------------------------------------------------------------


class SchemaDefinition(BaseModel):
    """
    The root model: describes the **entire** database schema to be processed.

    Invariant: ``table_map`` is an O(1) lookup cache built automatically
    from the ``tables`` list upon construction.
    """

    model_config = _SHARED_CONFIG

    tables: List[TableInfo] = Field(
        ..., min_length=1, description="All tables in the schema."
    )
    enums: List[EnumDefinition] = Field(
        default_factory=list, description="Standalone enum types."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (parsed from SQL comments, etc.).",
    )
    source_file: Optional[str] = Field(
        default=None, description="Original schema file path."
    )
    parsed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when schema was parsed."
    )

    # -- Internal cache (not part of the serialised model) ------------------
    _table_map: Dict[str, TableInfo] = {}

    @model_validator(mode="after")
    def _build_table_map(self) -> "SchemaDefinition":
        object.__setattr__(
            self,
            "_table_map",
            {t.name: t for t in self.tables},
        )
        return self

    @model_validator(mode="after")
    def _validate_unique_table_names(self) -> "SchemaDefinition":
        names: List[str] = [t.name for t in self.tables]
        if len(names) != len(set(names)):
            dupes: List[str] = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate table names: {sorted(set(dupes))}")
        return self

    @model_validator(mode="after")
    def _validate_fk_targets_exist(self) -> "SchemaDefinition":
        table_names: Set[str] = {t.name for t in self.tables}
        for table in self.tables:
            for fk in table.foreign_keys:
                if fk.referred_table not in table_names:
                    raise ValueError(
                        f"Table '{table.name}' has FK to '{fk.referred_table}' "
                        f"which is not defined in the schema."
                    )
        return self

    def get_table(self, name: str) -> Optional[TableInfo]:
        """O(1) table lookup."""
        return self._table_map.get(name)

    @computed_field  # type: ignore[misc]
    @property
    def table_count(self) -> int:
        return len(self.tables)

    @computed_field  # type: ignore[misc]
    @property
    def total_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    @computed_field  # type: ignore[misc]
    @property
    def total_relationships(self) -> int:
        return sum(len(t.relationships) for t in self.tables)

    @computed_field  # type: ignore[misc]
    @property
    def table_names(self) -> List[str]:
        return [t.name for t in self.tables]

    def topological_order(self) -> List[str]:
        """
        Return table names in dependency order (tables with no FKs first).

        Uses Kahn's algorithm — O(V + E) where V = tables, E = foreign keys.
        This ensures that generated migration files are in correct order.
        """
        in_degree: Dict[str, int] = {t.name: 0 for t in self.tables}
        adjacency: Dict[str, List[str]] = {t.name: [] for t in self.tables}

        for table in self.tables:
            for fk in table.foreign_keys:
                if fk.referred_table != table.name:  # skip self-referential
                    adjacency[fk.referred_table].append(table.name)
                    in_degree[table.name] += 1

        queue: List[str] = [n for n, d in in_degree.items() if d == 0]
        result: List[str] = []

        while queue:
            node: str = queue.pop(0)
            result.append(node)
            for neighbour in adjacency[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(result) != len(self.tables):
            logger.warning(
                "Circular FK dependency detected — topological sort is partial. "
                "Falling back to declaration order for remaining tables."
            )
            remaining: List[str] = [
                t.name for t in self.tables if t.name not in set(result)
            ]
            result.extend(remaining)

        return result

    def __repr__(self) -> str:
        return (
            f"<SchemaDefinition {self.table_count} tables, "
            f"{self.total_columns} columns, "
            f"{self.total_relationships} relationships>"
        )


# ---------------------------------------------------------------------------
# Generation Result — output manifest
# ---------------------------------------------------------------------------


class GeneratedFile(BaseModel):
    """Represents a single file produced by the code generator."""

    model_config = _SHARED_CONFIG

    path: str = Field(..., min_length=1, description="Relative file path.")
    content: str = Field(..., description="Full file content.")
    line_count: int = Field(default=0, ge=0, description="Number of lines.")
    size_bytes: int = Field(default=0, ge=0, description="Content size in bytes.")
    checksum: Optional[str] = Field(
        default=None, description="SHA-256 hex digest of content."
    )

    @model_validator(mode="after")
    def _compute_metrics(self) -> "GeneratedFile":
        self.line_count = self.content.count("\n") + (1 if self.content else 0)
        self.size_bytes = len(self.content.encode("utf-8"))
        return self


class GenerationResult(BaseModel):
    """
    Manifest returned by the generator after a full run.

    Consumed by exporters to write files to disk and by CLI to print
    summary statistics.
    """

    model_config = _SHARED_CONFIG

    files: List[GeneratedFile] = Field(
        default_factory=list, description="All generated files."
    )
    config: GenerationConfig = Field(
        ..., description="Config used for this generation run."
    )
    schema_def: SchemaDefinition = Field(
        ..., description="Schema that was processed."
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="Run start time."
    )
    finished_at: Optional[datetime] = Field(
        default=None, description="Run finish time."
    )
    success: bool = Field(default=True, description="Overall success flag.")
    errors: List[str] = Field(
        default_factory=list, description="Non-fatal errors / warnings."
    )

    @computed_field  # type: ignore[misc]
    @property
    def total_files(self) -> int:
        return len(self.files)

    @computed_field  # type: ignore[misc]
    @property
    def total_lines(self) -> int:
        return sum(f.line_count for f in self.files)

    @computed_field  # type: ignore[misc]
    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @computed_field  # type: ignore[misc]
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def add_file(self, path: str, content: str) -> None:
        """Append a generated file — O(1) amortised."""
        self.files.append(GeneratedFile(path=path, content=content))

    def __repr__(self) -> str:
        return (
            f"<GenerationResult {self.total_files} files, "
            f"{self.total_lines} lines, "
            f"{'OK' if self.success else 'FAILED'}>"
        )


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "ColumnType",
    "RelationshipType",
    "IndexType",
    "OnDeleteAction",
    "OnUpdateAction",
    "AuthStrategy",
    "PaginationStyle",
    "NamingConvention",
    "DatabaseDialect",
    "ColumnConstraint",
    "EnumDefinition",
    "ColumnDefault",
    "ColumnInfo",
    "ForeignKeyInfo",
    "RelationshipInfo",
    "IndexInfo",
    "UniqueConstraintInfo",
    "TableInfo",
    "CRUDConfig",
    "GenerationConfig",
    "SchemaDefinition",
    "GeneratedFile",
    "GenerationResult",
]

logger.debug("apigen.models loaded — %d public symbols.", len(__all__))
