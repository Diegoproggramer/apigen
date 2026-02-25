# File: apigen/validators.py
"""
NexaFlow APIGen - Schema & Configuration Validators
=====================================================
This module provides a **pure-function validation pipeline** that operates
on the Pydantic V2 models defined in ``apigen.models``.

Pydantic's built-in validators handle per-field and per-model structural
correctness.  This module adds **cross-entity semantic validation**: FK
target resolution, circular dependency detection, naming convention
compliance, configuration sanity checks, and more.

All functions are designed as O(n) single-pass algorithms where n is the
total number of entities (tables, columns, FKs, etc.).

Usage by downstream modules:
    from apigen.validators import validate_full
    errors = validate_full(schema_def, generation_config)
    if errors:
        raise SystemExit(...)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from apigen.models import (
    AuthStrategy,
    ColumnInfo,
    ColumnType,
    CRUDConfig,
    DatabaseDialect,
    EnumDefinition,
    ForeignKeyInfo,
    GenerationConfig,
    IndexInfo,
    NamingConvention,
    RelationshipInfo,
    RelationshipType,
    SchemaDefinition,
    TableInfo,
    UniqueConstraintInfo,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen.validators")

# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------


class ValidationError:
    """Lightweight error descriptor (no Pydantic overhead)."""

    __slots__ = ("level", "code", "message", "context")

    def __init__(
        self,
        level: str,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.level: str = level  # "error" | "warning" | "info"
        self.code: str = code
        self.message: str = message
        self.context: Dict[str, Any] = context or {}

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    @property
    def is_warning(self) -> bool:
        return self.level == "warning"

    def __repr__(self) -> str:
        return f"[{self.level.upper()}] {self.code}: {self.message}"

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class ValidationResult:
    """
    Accumulates ``ValidationError`` instances produced by the pipeline.

    Provides O(1) access to counts and O(n) filtering.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: List[ValidationError] = []

    # -- Mutation -----------------------------------------------------------

    def add_error(
        self,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._items.append(ValidationError("error", code, message, context))

    def add_warning(
        self,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._items.append(ValidationError("warning", code, message, context))

    def add_info(
        self,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._items.append(ValidationError("info", code, message, context))

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one — O(k) where k = len(other)."""
        self._items.extend(other._items)

    # -- Query --------------------------------------------------------------

    @property
    def errors(self) -> List[ValidationError]:
        return [e for e in self._items if e.is_error]

    @property
    def warnings(self) -> List[ValidationError]:
        return [e for e in self._items if e.is_warning]

    @property
    def all_items(self) -> List[ValidationError]:
        return list(self._items)

    @property
    def has_errors(self) -> bool:
        return any(e.is_error for e in self._items)

    @property
    def has_warnings(self) -> bool:
        return any(e.is_warning for e in self._items)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self._items if e.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self._items if e.is_warning)

    @property
    def is_valid(self) -> bool:
        return not self.has_errors

    def summary(self) -> str:
        return (
            f"Validation: {self.error_count} error(s), "
            f"{self.warning_count} warning(s), "
            f"{len(self._items)} total item(s)."
        )

    def __repr__(self) -> str:
        return f"<ValidationResult {self.summary()}>"

    def __bool__(self) -> bool:
        """Truthy when there are NO errors (i.e. valid)."""
        return self.is_valid

    def __len__(self) -> int:
        return len(self._items)

    def format_report(self, include_info: bool = False) -> str:
        """Human-readable multi-line report."""
        lines: List[str] = [self.summary(), ""]
        for item in self._items:
            if not include_info and item.level == "info":
                continue
            prefix: str = {
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
            }.get(item.level, "•")
            lines.append(f"  {prefix} [{item.code}] {item.message}")
            if item.context:
                for k, v in item.context.items():
                    lines.append(f"       {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at module load — O(1) per use)
# ---------------------------------------------------------------------------

_SNAKE_CASE_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
_PASCAL_CASE_RE: re.Pattern[str] = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_IDENTIFIER_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_SEMANTIC_VERSION_RE: re.Pattern[str] = re.compile(
    r"^\d+\.\d+\.\d+([a-zA-Z0-9\.\-]+)?$"
)

# SQL reserved words that must not be used as identifiers (subset of most common)
_SQL_RESERVED_WORDS: FrozenSet[str] = frozenset(
    {
        "select", "insert", "update", "delete", "drop", "create", "alter",
        "table", "column", "index", "from", "where", "join", "inner",
        "outer", "left", "right", "on", "and", "or", "not", "null",
        "true", "false", "in", "between", "like", "is", "as", "order",
        "by", "group", "having", "limit", "offset", "union", "all",
        "distinct", "case", "when", "then", "else", "end", "exists",
        "primary", "foreign", "key", "references", "constraint", "check",
        "default", "unique", "cascade", "set", "values", "into",
        "grant", "revoke", "begin", "commit", "rollback", "transaction",
        "user", "role", "schema", "database", "trigger", "procedure",
        "function", "view", "sequence", "type", "enum", "domain",
        "return", "returns", "declare", "execute", "fetch", "cursor",
        "open", "close", "deallocate", "prepare", "with", "recursive",
    }
)

# Python reserved / built-in names to avoid in generated code
_PYTHON_RESERVED_WORDS: FrozenSet[str] = frozenset(
    {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else",
        "except", "finally", "for", "from", "global", "if", "import",
        "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
        "return", "try", "while", "with", "yield",
        "id", "type", "list", "dict", "set", "str", "int", "float",
        "bool", "bytes", "object", "print", "input", "range", "len",
        "map", "filter", "zip", "enumerate", "super", "property",
        "staticmethod", "classmethod", "hash", "help", "dir", "vars",
        "getattr", "setattr", "delattr", "hasattr", "isinstance",
        "issubclass", "callable", "repr", "format", "iter", "next",
        "open", "file", "exec", "eval", "compile",
    }
)

# Dialect-specific type restrictions
_DIALECT_UNSUPPORTED_TYPES: Dict[str, FrozenSet[str]] = {
    "sqlite": frozenset({"jsonb", "hstore", "inet", "cidr", "macaddr", "array"}),
    "mysql": frozenset({"jsonb", "hstore", "inet", "cidr", "macaddr"}),
    "mssql": frozenset({"jsonb", "hstore", "inet", "cidr", "macaddr", "array"}),
    "oracle": frozenset({"jsonb", "hstore", "inet", "cidr", "macaddr", "array"}),
    "postgresql": frozenset(),  # supports everything
}


# ---------------------------------------------------------------------------
# Individual validation functions (each is O(n) or better)
# ---------------------------------------------------------------------------


def validate_table_names(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate all table names for:
    - Valid Python / SQL identifier format
    - snake_case convention
    - No SQL reserved words
    - No Python reserved words
    - No duplicates (also checked by Pydantic, belt-and-suspenders)

    Complexity: O(T) where T = number of tables.
    """
    result: ValidationResult = ValidationResult()
    seen: Set[str] = set()

    for table in schema.tables:
        name: str = table.name
        ctx: Dict[str, Any] = {"table": name}

        # Duplicate check
        if name in seen:
            result.add_error(
                "DUPLICATE_TABLE_NAME",
                f"Table name '{name}' is defined more than once.",
                ctx,
            )
        seen.add(name)

        # Valid identifier
        if not _IDENTIFIER_RE.match(name):
            result.add_error(
                "INVALID_TABLE_NAME",
                f"Table name '{name}' is not a valid identifier.",
                ctx,
            )
            continue  # skip further checks for this name

        # Snake case
        if not _SNAKE_CASE_RE.match(name):
            result.add_warning(
                "TABLE_NAME_NOT_SNAKE_CASE",
                f"Table name '{name}' is not snake_case. "
                f"Generated class name may look odd.",
                ctx,
            )

        # SQL reserved
        if name.lower() in _SQL_RESERVED_WORDS:
            result.add_error(
                "TABLE_NAME_SQL_RESERVED",
                f"Table name '{name}' is a SQL reserved word.",
                ctx,
            )

        # Python reserved
        if name in _PYTHON_RESERVED_WORDS:
            result.add_error(
                "TABLE_NAME_PYTHON_RESERVED",
                f"Table name '{name}' clashes with a Python reserved word.",
                ctx,
            )

    logger.debug(
        "validate_table_names: checked %d tables, %d issue(s).",
        len(schema.tables),
        len(result),
    )
    return result


def validate_column_names(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate every column name across all tables.

    Complexity: O(C) where C = total number of columns.
    """
    result: ValidationResult = ValidationResult()

    for table in schema.tables:
        col_names_seen: Set[str] = set()
        for col in table.columns:
            ctx: Dict[str, Any] = {"table": table.name, "column": col.name}

            # Duplicate within table
            if col.name in col_names_seen:
                result.add_error(
                    "DUPLICATE_COLUMN_NAME",
                    f"Column '{col.name}' is duplicated in table '{table.name}'.",
                    ctx,
                )
            col_names_seen.add(col.name)

            # Valid identifier
            if not _IDENTIFIER_RE.match(col.name):
                result.add_error(
                    "INVALID_COLUMN_NAME",
                    f"Column '{col.name}' in table '{table.name}' "
                    f"is not a valid identifier.",
                    ctx,
                )
                continue

            # Snake case
            if not _SNAKE_CASE_RE.match(col.name):
                result.add_warning(
                    "COLUMN_NAME_NOT_SNAKE_CASE",
                    f"Column '{col.name}' in table '{table.name}' "
                    f"is not snake_case.",
                    ctx,
                )

            # Python reserved
            if col.name in _PYTHON_RESERVED_WORDS:
                result.add_warning(
                    "COLUMN_NAME_PYTHON_RESERVED",
                    f"Column '{col.name}' in table '{table.name}' "
                    f"clashes with Python built-in. "
                    f"Generated code will use '{col.name}_' suffix.",
                    ctx,
                )

            # SQL reserved
            if col.name.lower() in _SQL_RESERVED_WORDS:
                result.add_warning(
                    "COLUMN_NAME_SQL_RESERVED",
                    f"Column '{col.name}' in table '{table.name}' "
                    f"is a SQL reserved word. Quoting will be applied.",
                    ctx,
                )

    logger.debug("validate_column_names: completed for %d tables.", len(schema.tables))
    return result


def validate_primary_keys(schema: SchemaDefinition) -> ValidationResult:
    """
    Ensure every table has at least one primary key column.

    Complexity: O(T) where T = number of tables.
    """
    result: ValidationResult = ValidationResult()

    for table in schema.tables:
        pk_cols: List[str] = table.resolved_primary_keys
        ctx: Dict[str, Any] = {"table": table.name}

        if not pk_cols:
            result.add_error(
                "MISSING_PRIMARY_KEY",
                f"Table '{table.name}' has no primary key. "
                f"Every generated ORM model requires at least one PK column.",
                ctx,
            )
            continue

        # Verify PK columns actually exist in the table
        col_set: Set[str] = {c.name for c in table.columns}
        for pk in pk_cols:
            if pk not in col_set:
                result.add_error(
                    "PK_COLUMN_NOT_FOUND",
                    f"Primary key column '{pk}' declared for table "
                    f"'{table.name}' does not exist in columns list.",
                    {"table": table.name, "pk_column": pk},
                )

        # Warn on nullable PK
        for col in table.columns:
            if col.primary_key and col.nullable:
                result.add_warning(
                    "NULLABLE_PRIMARY_KEY",
                    f"PK column '{col.name}' in table '{table.name}' "
                    f"is marked nullable — this is usually a mistake.",
                    {"table": table.name, "column": col.name},
                )

    return result


def validate_foreign_keys(schema: SchemaDefinition) -> ValidationResult:
    """
    Cross-table FK validation:
    - Referred table exists
    - Referred column exists in referred table
    - Constrained column type is compatible with referred column type
    - No self-referential FK to a non-existent column

    Complexity: O(T + F) where F = total foreign keys.
    """
    result: ValidationResult = ValidationResult()

    # Build global column type map: (table_name, col_name) → ColumnType   O(C)
    col_type_map: Dict[Tuple[str, str], ColumnType] = {}
    for table in schema.tables:
        for col in table.columns:
            col_type_map[(table.name, col.name)] = col.column_type

    table_names: Set[str] = {t.name for t in schema.tables}

    for table in schema.tables:
        for fk in table.foreign_keys:
            ctx: Dict[str, Any] = {
                "table": table.name,
                "column": fk.constrained_column,
                "referred_table": fk.referred_table,
                "referred_column": fk.referred_column,
            }

            # Target table exists (also checked by Pydantic — belt-and-suspenders)
            if fk.referred_table not in table_names:
                result.add_error(
                    "FK_TARGET_TABLE_MISSING",
                    f"FK from '{table.name}.{fk.constrained_column}' → "
                    f"'{fk.referred_table}.{fk.referred_column}': "
                    f"table '{fk.referred_table}' does not exist.",
                    ctx,
                )
                continue

            # Target column exists
            target_key: Tuple[str, str] = (fk.referred_table, fk.referred_column)
            if target_key not in col_type_map:
                result.add_error(
                    "FK_TARGET_COLUMN_MISSING",
                    f"FK from '{table.name}.{fk.constrained_column}' → "
                    f"'{fk.referred_table}.{fk.referred_column}': "
                    f"column '{fk.referred_column}' not found in "
                    f"'{fk.referred_table}'.",
                    ctx,
                )
                continue

            # Type compatibility check
            source_key: Tuple[str, str] = (table.name, fk.constrained_column)
            source_type: Optional[ColumnType] = col_type_map.get(source_key)
            target_type: ColumnType = col_type_map[target_key]

            if source_type is not None and source_type != target_type:
                # Allow compatible pairings (e.g. integer ↔ biginteger)
                compatible_groups: List[FrozenSet[ColumnType]] = [
                    frozenset({
                        ColumnType.INTEGER,
                        ColumnType.BIGINTEGER,
                        ColumnType.SMALLINTEGER,
                    }),
                    frozenset({
                        ColumnType.STRING,
                        ColumnType.VARCHAR,
                        ColumnType.CHAR,
                        ColumnType.TEXT,
                    }),
                    frozenset({
                        ColumnType.FLOAT,
                        ColumnType.DOUBLE,
                        ColumnType.NUMERIC,
                    }),
                    frozenset({
                        ColumnType.DATETIME,
                        ColumnType.TIMESTAMP,
                    }),
                ]
                is_compatible: bool = any(
                    source_type in group and target_type in group
                    for group in compatible_groups
                )
                if not is_compatible:
                    result.add_warning(
                        "FK_TYPE_MISMATCH",
                        f"FK '{table.name}.{fk.constrained_column}' "
                        f"(type={source_type}) → "
                        f"'{fk.referred_table}.{fk.referred_column}' "
                        f"(type={target_type}): types may be incompatible.",
                        ctx,
                    )

    return result


def validate_relationships(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate ORM relationship definitions:
    - Target table exists
    - Underlying FK is valid
    - M2M relationships have a secondary table
    - back_populates targets are consistent

    Complexity: O(T + R) where R = total relationships.
    """
    result: ValidationResult = ValidationResult()
    table_names: Set[str] = {t.name for t in schema.tables}

    # Build reverse map: (table_name, rel_name) → RelationshipInfo    O(R)
    rel_map: Dict[Tuple[str, str], RelationshipInfo] = {}
    for table in schema.tables:
        for rel in table.relationships:
            rel_map[(table.name, rel.name)] = rel

    for table in schema.tables:
        for rel in table.relationships:
            ctx: Dict[str, Any] = {
                "table": table.name,
                "relationship": rel.name,
                "target": rel.target_table,
            }

            # Target table
            if rel.target_table not in table_names:
                result.add_error(
                    "REL_TARGET_TABLE_MISSING",
                    f"Relationship '{rel.name}' on table '{table.name}' "
                    f"targets non-existent table '{rel.target_table}'.",
                    ctx,
                )
                continue

            # M2M must have secondary
            if (
                rel.relationship_type == RelationshipType.MANY_TO_MANY
                and not rel.secondary_table
            ):
                result.add_error(
                    "M2M_MISSING_SECONDARY",
                    f"Many-to-many relationship '{rel.name}' on "
                    f"'{table.name}' is missing 'secondary_table'.",
                    ctx,
                )

            # If secondary_table is specified, it must exist
            if rel.secondary_table and rel.secondary_table not in table_names:
                result.add_error(
                    "REL_SECONDARY_TABLE_MISSING",
                    f"Relationship '{rel.name}' on '{table.name}' "
                    f"references secondary table '{rel.secondary_table}' "
                    f"which does not exist.",
                    ctx,
                )

            # back_populates consistency
            if rel.back_populates:
                reverse_key: Tuple[str, str] = (
                    rel.target_table,
                    rel.back_populates,
                )
                reverse_rel: Optional[RelationshipInfo] = rel_map.get(reverse_key)
                if reverse_rel is None:
                    result.add_warning(
                        "REL_BACK_POPULATES_MISSING",
                        f"Relationship '{rel.name}' on '{table.name}' "
                        f"sets back_populates='{rel.back_populates}' but "
                        f"no such relationship exists on '{rel.target_table}'.",
                        ctx,
                    )
                elif reverse_rel.back_populates != rel.name:
                    result.add_warning(
                        "REL_BACK_POPULATES_MISMATCH",
                        f"Relationship '{rel.name}' on '{table.name}' "
                        f"(back_populates='{rel.back_populates}') and "
                        f"'{reverse_rel.name}' on '{rel.target_table}' "
                        f"(back_populates='{reverse_rel.back_populates}') "
                        f"are not symmetrical.",
                        ctx,
                    )

    return result


def validate_indexes(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate index definitions across all tables.

    Complexity: O(T + I) where I = total indexes.
    """
    result: ValidationResult = ValidationResult()

    for table in schema.tables:
        idx_names_seen: Set[str] = set()
        col_set: Set[str] = {c.name for c in table.columns}

        for idx in table.indexes:
            ctx: Dict[str, Any] = {
                "table": table.name,
                "index": idx.name,
            }

            # Duplicate index name within table
            if idx.name in idx_names_seen:
                result.add_warning(
                    "DUPLICATE_INDEX_NAME",
                    f"Index name '{idx.name}' appears more than once "
                    f"in table '{table.name}'.",
                    ctx,
                )
            idx_names_seen.add(idx.name)

            # Column existence (also checked by Pydantic model_validator)
            missing: List[str] = [c for c in idx.columns if c not in col_set]
            if missing:
                result.add_error(
                    "INDEX_COLUMN_MISSING",
                    f"Index '{idx.name}' on '{table.name}' references "
                    f"non-existent column(s): {missing}.",
                    ctx,
                )

            # Single-column unique index on a column already marked unique
            if idx.unique and len(idx.columns) == 1:
                col_name: str = idx.columns[0]
                col_info: Optional[ColumnInfo] = table.get_column(col_name)
                if col_info and col_info.unique:
                    result.add_info(
                        "REDUNDANT_UNIQUE_INDEX",
                        f"Index '{idx.name}' is a unique index on column "
                        f"'{col_name}' which is already marked unique. "
                        f"This creates a redundant constraint.",
                        ctx,
                    )

    return result


def validate_circular_dependencies(schema: SchemaDefinition) -> ValidationResult:
    """
    Detect circular FK dependency chains using iterative DFS.

    Complexity: O(T + F) — standard cycle detection.
    """
    result: ValidationResult = ValidationResult()

    # Adjacency list: table → set of tables it depends on
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for table in schema.tables:
        for fk in table.foreign_keys:
            if fk.referred_table != table.name:  # ignore self-ref for cycles
                adjacency[table.name].add(fk.referred_table)

    all_tables: Set[str] = {t.name for t in schema.tables}
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    cycles_found: List[List[str]] = []

    for start in all_tables:
        if start in visited:
            continue

        # Iterative DFS using an explicit stack
        stack: List[Tuple[str, bool]] = [(start, False)]
        path: List[str] = []

        while stack:
            node, is_returning = stack.pop()

            if is_returning:
                in_stack.discard(node)
                if path and path[-1] == node:
                    path.pop()
                continue

            if node in in_stack:
                # Found a cycle — extract it
                cycle_start_idx: int = (
                    path.index(node) if node in path else len(path)
                )
                cycle: List[str] = path[cycle_start_idx:] + [node]
                cycles_found.append(cycle)
                continue

            if node in visited:
                continue

            visited.add(node)
            in_stack.add(node)
            path.append(node)

            # Push return marker then neighbours
            stack.append((node, True))
            for neighbour in adjacency.get(node, set()):
                stack.append((neighbour, False))

    for cycle in cycles_found:
        cycle_str: str = " → ".join(cycle)
        result.add_warning(
            "CIRCULAR_FK_DEPENDENCY",
            f"Circular foreign-key dependency detected: {cycle_str}. "
            f"Generated Alembic migrations may need manual ordering.",
            {"cycle": cycle},
        )

    if not cycles_found:
        logger.debug("No circular FK dependencies detected.")

    return result


def validate_dialect_compatibility(
    schema: SchemaDefinition, config: GenerationConfig
) -> ValidationResult:
    """
    Check that all column types used are supported by the target dialect.

    Complexity: O(C) where C = total columns.
    """
    result: ValidationResult = ValidationResult()
    dialect: str = config.database_dialect
    unsupported: FrozenSet[str] = _DIALECT_UNSUPPORTED_TYPES.get(
        dialect, frozenset()
    )

    if not unsupported:
        return result  # nothing to check (e.g. PostgreSQL)

    for table in schema.tables:
        for col in table.columns:
            if col.column_type in unsupported:
                result.add_error(
                    "DIALECT_UNSUPPORTED_TYPE",
                    f"Column '{col.name}' in table '{table.name}' uses "
                    f"type '{col.column_type}' which is not supported by "
                    f"the '{dialect}' dialect.",
                    {
                        "table": table.name,
                        "column": col.name,
                        "type": col.column_type,
                        "dialect": dialect,
                    },
                )

    return result


def validate_enum_definitions(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate standalone enums and column-level enum definitions.

    Complexity: O(E + C) where E = standalone enums, C = columns.
    """
    result: ValidationResult = ValidationResult()

    # Standalone enums: unique names
    enum_names_seen: Set[str] = set()
    for enum_def in schema.enums:
        if enum_def.name in enum_names_seen:
            result.add_error(
                "DUPLICATE_ENUM_NAME",
                f"Enum type '{enum_def.name}' is defined more than once.",
                {"enum": enum_def.name},
            )
        enum_names_seen.add(enum_def.name)

        # Identifier check
        if not _IDENTIFIER_RE.match(enum_def.name):
            result.add_error(
                "INVALID_ENUM_NAME",
                f"Enum name '{enum_def.name}' is not a valid identifier.",
                {"enum": enum_def.name},
            )

        # At least 1 value (also checked by Pydantic)
        if not enum_def.values:
            result.add_error(
                "EMPTY_ENUM",
                f"Enum '{enum_def.name}' has no values.",
                {"enum": enum_def.name},
            )

    # Column-level enum checks
    for table in schema.tables:
        for col in table.columns:
            if col.column_type == ColumnType.ENUM and col.enum_definition:
                if not col.enum_definition.values:
                    result.add_error(
                        "EMPTY_COLUMN_ENUM",
                        f"Column '{col.name}' in table '{table.name}' is "
                        f"of type ENUM but has no values defined.",
                        {"table": table.name, "column": col.name},
                    )

    return result


def validate_generation_config(config: GenerationConfig) -> ValidationResult:
    """
    Validate the generation configuration model for semantic correctness
    beyond what Pydantic field constraints enforce.

    Complexity: O(1) (config is a fixed-size object).
    """
    result: ValidationResult = ValidationResult()

    # Project name is a valid Python package name
    if not _IDENTIFIER_RE.match(config.project_name):
        result.add_error(
            "INVALID_PROJECT_NAME",
            f"Project name '{config.project_name}' is not a valid "
            f"Python identifier / package name.",
            {"project_name": config.project_name},
        )

    # Version looks like semver
    if not _SEMANTIC_VERSION_RE.match(config.project_version):
        result.add_warning(
            "INVALID_SEMVER",
            f"Project version '{config.project_version}' does not follow "
            f"semantic versioning (e.g. 1.0.0).",
            {"version": config.project_version},
        )

    # Async driver check for known dialects
    if config.use_async:
        async_driver_hints: Dict[str, str] = {
            "postgresql": "asyncpg",
            "mysql": "aiomysql",
            "sqlite": "aiosqlite",
        }
        dialect: str = config.database_dialect
        hint: Optional[str] = async_driver_hints.get(dialect)
        if hint and hint not in config.database_url:
            result.add_warning(
                "ASYNC_DRIVER_MISSING",
                f"use_async=True but database_url does not contain "
                f"'{hint}' (recommended async driver for {dialect}). "
                f"Current URL: {config.database_url}",
                {"dialect": dialect, "suggested_driver": hint},
            )

    # API prefix should start with /
    if not config.api_prefix.startswith("/"):
        result.add_warning(
            "API_PREFIX_NO_SLASH",
            f"api_prefix '{config.api_prefix}' should start with '/'.",
            {"api_prefix": config.api_prefix},
        )

    # CORS origins
    if config.enable_cors and not config.cors_origins:
        result.add_warning(
            "CORS_NO_ORIGINS",
            "CORS is enabled but cors_origins list is empty.",
        )

    # Rate limiting
    if config.enable_rate_limiting and config.rate_limit_per_minute < 10:
        result.add_warning(
            "RATE_LIMIT_TOO_LOW",
            f"Rate limit of {config.rate_limit_per_minute}/min is very low "
            f"and may cause issues during development.",
            {"rate_limit": config.rate_limit_per_minute},
        )

    # Output dir
    if not config.output_dir:
        result.add_error(
            "EMPTY_OUTPUT_DIR",
            "output_dir must not be empty.",
        )

    return result


def validate_column_constraints(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate column-level semantic constraints:
    - max_length set only for string types
    - precision/scale set only for numeric types
    - autoincrement only on integer PK columns
    - enum columns have definitions

    Complexity: O(C) total columns.
    """
    result: ValidationResult = ValidationResult()

    _STRING_TYPES: FrozenSet[ColumnType] = frozenset(
        {ColumnType.STRING, ColumnType.VARCHAR, ColumnType.CHAR}
    )
    _NUMERIC_PRECISION_TYPES: FrozenSet[ColumnType] = frozenset(
        {ColumnType.NUMERIC, ColumnType.DOUBLE, ColumnType.FLOAT}
    )
    _INTEGER_TYPES: FrozenSet[ColumnType] = frozenset(
        {ColumnType.INTEGER, ColumnType.BIGINTEGER, ColumnType.SMALLINTEGER}
    )

    for table in schema.tables:
        for col in table.columns:
            ctx: Dict[str, Any] = {"table": table.name, "column": col.name}

            # max_length on non-string
            if col.max_length is not None and col.column_type not in _STRING_TYPES:
                result.add_warning(
                    "MAX_LENGTH_ON_NON_STRING",
                    f"Column '{col.name}' in '{table.name}' has max_length "
                    f"but type is '{col.column_type}' — max_length will be "
                    f"ignored.",
                    ctx,
                )

            # precision on non-numeric
            if (
                col.precision is not None
                and col.column_type not in _NUMERIC_PRECISION_TYPES
            ):
                result.add_warning(
                    "PRECISION_ON_NON_NUMERIC",
                    f"Column '{col.name}' in '{table.name}' has precision "
                    f"but type is '{col.column_type}'.",
                    ctx,
                )

            # autoincrement on non-integer
            if col.autoincrement and col.column_type not in _INTEGER_TYPES:
                result.add_error(
                    "AUTOINCREMENT_NON_INTEGER",
                    f"Column '{col.name}' in '{table.name}' is "
                    f"autoincrement but type is '{col.column_type}'. "
                    f"Autoincrement is only valid for integer types.",
                    ctx,
                )

            # autoincrement without PK
            if col.autoincrement and not col.primary_key:
                result.add_warning(
                    "AUTOINCREMENT_NON_PK",
                    f"Column '{col.name}' in '{table.name}' is autoincrement "
                    f"but not a primary key. This may cause issues with some "
                    f"databases.",
                    ctx,
                )

            # Very large VARCHAR without need
            if (
                col.column_type in _STRING_TYPES
                and col.max_length is not None
                and col.max_length > 10000
            ):
                result.add_warning(
                    "VERY_LARGE_VARCHAR",
                    f"Column '{col.name}' in '{table.name}' has max_length "
                    f"of {col.max_length}. Consider using TEXT instead.",
                    ctx,
                )

    return result


def validate_unique_constraints(schema: SchemaDefinition) -> ValidationResult:
    """
    Validate multi-column unique constraints.

    Complexity: O(U * k) where U = unique constraints, k = avg columns per constraint.
    """
    result: ValidationResult = ValidationResult()

    for table in schema.tables:
        col_set: Set[str] = {c.name for c in table.columns}

        for uc in table.unique_constraints:
            ctx: Dict[str, Any] = {
                "table": table.name,
                "constraint_name": uc.name or "(unnamed)",
                "columns": uc.columns,
            }

            missing: List[str] = [c for c in uc.columns if c not in col_set]
            if missing:
                result.add_error(
                    "UNIQUE_CONSTRAINT_COLUMN_MISSING",
                    f"Unique constraint on '{table.name}' references "
                    f"non-existent column(s): {missing}.",
                    ctx,
                )

            # Single-column UC that duplicates column.unique flag
            if len(uc.columns) == 1:
                col_obj: Optional[ColumnInfo] = table.get_column(uc.columns[0])
                if col_obj and col_obj.unique:
                    result.add_info(
                        "REDUNDANT_UNIQUE_CONSTRAINT",
                        f"Unique constraint on '{table.name}' for column "
                        f"'{uc.columns[0]}' is redundant — column already "
                        f"marked unique.",
                        ctx,
                    )

    return result


def validate_schema_size(schema: SchemaDefinition) -> ValidationResult:
    """
    Emit informational messages about schema size for performance awareness.

    Complexity: O(1).
    """
    result: ValidationResult = ValidationResult()

    if schema.table_count > 200:
        result.add_warning(
            "LARGE_SCHEMA",
            f"Schema contains {schema.table_count} tables. Generation "
            f"may take longer than usual. Consider splitting into modules.",
            {"table_count": schema.table_count},
        )

    if schema.total_columns > 5000:
        result.add_warning(
            "VERY_MANY_COLUMNS",
            f"Schema has {schema.total_columns} total columns across all "
            f"tables.",
            {"total_columns": schema.total_columns},
        )

    result.add_info(
        "SCHEMA_STATS",
        f"Schema: {schema.table_count} tables, "
        f"{schema.total_columns} columns, "
        f"{schema.total_relationships} relationships.",
        {
            "tables": schema.table_count,
            "columns": schema.total_columns,
            "relationships": schema.total_relationships,
        },
    )

    return result


# ---------------------------------------------------------------------------
# Composite validation orchestrators
# ---------------------------------------------------------------------------

# Type alias for a validation function
ValidatorFn = Callable[..., ValidationResult]


def validate_schema(schema: SchemaDefinition) -> ValidationResult:
    """
    Run all schema-level validators.  Returns a merged ``ValidationResult``.

    Complexity: O(T + C + F + R + I + U) — linear in total entities.
    """
    result: ValidationResult = ValidationResult()

    validators: List[Callable[[SchemaDefinition], ValidationResult]] = [
        validate_table_names,
        validate_column_names,
        validate_primary_keys,
        validate_foreign_keys,
        validate_relationships,
        validate_indexes,
        validate_circular_dependencies,
        validate_enum_definitions,
        validate_column_constraints,
        validate_unique_constraints,
        validate_schema_size,
    ]

    for validator_fn in validators:
        logger.debug("Running validator: %s", validator_fn.__name__)
        sub_result: ValidationResult = validator_fn(schema)
        result.merge(sub_result)

    logger.info("Schema validation complete: %s", result.summary())
    return result


def validate_config(config: GenerationConfig) -> ValidationResult:
    """
    Run all configuration-level validators.

    Complexity: O(1).
    """
    result: ValidationResult = validate_generation_config(config)
    logger.info("Config validation complete: %s", result.summary())
    return result


def validate_full(
    schema: SchemaDefinition,
    config: GenerationConfig,
) -> ValidationResult:
    """
    **Master validation entry point.**

    Runs all schema validators, all config validators, and cross-cutting
    validators (e.g. dialect compatibility).

    This is the single function that ``generator.py`` and ``cli.py`` call
    before starting code generation.

    Complexity: O(T + C + F + R + I + U) — linear in total schema entities.
    """
    logger.info(
        "Starting full validation — %d tables, dialect=%s",
        schema.table_count,
        config.database_dialect,
    )

    result: ValidationResult = ValidationResult()

    # Schema-level
    result.merge(validate_schema(schema))

    # Config-level
    result.merge(validate_config(config))

    # Cross-cutting: dialect compatibility
    result.merge(validate_dialect_compatibility(schema, config))

    # Cross-cutting: CRUD overrides reference valid tables
    table_names: Set[str] = {t.name for t in schema.tables}
    for override_table in config.crud_overrides:
        if override_table not in table_names:
            result.add_warning(
                "CRUD_OVERRIDE_UNKNOWN_TABLE",
                f"CRUD override defined for table '{override_table}' "
                f"which does not exist in the schema.",
                {"table": override_table},
            )

    if result.has_errors:
        logger.error(
            "Validation FAILED with %d error(s). %s",
            result.error_count,
            result.summary(),
        )
    else:
        logger.info("Validation PASSED. %s", result.summary())

    return result


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "ValidationError",
    "ValidationResult",
    "validate_table_names",
    "validate_column_names",
    "validate_primary_keys",
    "validate_foreign_keys",
    "validate_relationships",
    "validate_indexes",
    "validate_circular_dependencies",
    "validate_dialect_compatibility",
    "validate_enum_definitions",
    "validate_generation_config",
    "validate_column_constraints",
    "validate_unique_constraints",
    "validate_schema_size",
    "validate_schema",
    "validate_config",
    "validate_full",
]

logger.debug("apigen.validators loaded — %d public symbols.", len(__all__))
