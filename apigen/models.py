# File: apigen/models.py
"""
NexaFlow APIGen - Model Definitions
====================================
Enterprise-grade Pydantic V2 models for database schema representation.
Generates SQLAlchemy ORM models and Pydantic schemas from definitions.

Performance: O(n) generation with pre-computed type mappings.
"""

from __future__ import annotations

import re
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field as dataclass_field

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================
# LOGGING SETUP
# ============================================================

logger = logging.getLogger("apigen.models")


# ============================================================
# ENUMERATIONS
# ============================================================

class FieldType(str, Enum):
    """
    Supported database field types.
    Using str mixin for direct JSON serialization and string comparison.
    """
    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    JSON = "json"
    ENUM = "enum"
    FILE = "file"
    IMAGE = "image"
    PASSWORD = "password"
    PHONE = "phone"
    SLUG = "slug"
    IP_ADDRESS = "ip_address"
    BINARY = "binary"


class RelationType(str, Enum):
    """Database relationship cardinality types."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class IndexType(str, Enum):
    """Database index types."""
    NORMAL = "index"
    UNIQUE = "unique"
    COMPOSITE = "composite"
    FULLTEXT = "fulltext"


class OnDelete(str, Enum):
    """Foreign key ON DELETE referential actions."""
    CASCADE = "CASCADE"
    SET_NULL = "SET NULL"
    RESTRICT = "RESTRICT"
    NO_ACTION = "NO ACTION"
    SET_DEFAULT = "SET DEFAULT"


# ============================================================
# PRE-COMPUTED TYPE MAPPINGS (O(1) lookup)
# ============================================================

SQLALCHEMY_TYPE_MAP: Dict[FieldType, str] = {
    FieldType.STRING: "String({max_length})",
    FieldType.TEXT: "Text",
    FieldType.INTEGER: "Integer",
    FieldType.FLOAT: "Float",
    FieldType.DECIMAL: "Numeric(precision={precision}, scale={scale})",
    FieldType.BOOLEAN: "Boolean",
    FieldType.DATETIME: "DateTime(timezone=True)",
    FieldType.DATE: "Date",
    FieldType.TIME: "Time",
    FieldType.EMAIL: "String(320)",
    FieldType.URL: "String(2048)",
    FieldType.UUID: "String(36)",
    FieldType.JSON: "JSON",
    FieldType.ENUM: "String(50)",
    FieldType.FILE: "String(512)",
    FieldType.IMAGE: "String(512)",
    FieldType.PASSWORD: "String(128)",
    FieldType.PHONE: "String(20)",
    FieldType.SLUG: "String({max_length})",
    FieldType.IP_ADDRESS: "String(45)",
    FieldType.BINARY: "LargeBinary",
}

PYDANTIC_TYPE_MAP: Dict[FieldType, str] = {
    FieldType.STRING: "str",
    FieldType.TEXT: "str",
    FieldType.INTEGER: "int",
    FieldType.FLOAT: "float",
    FieldType.DECIMAL: "Decimal",
    FieldType.BOOLEAN: "bool",
    FieldType.DATETIME: "datetime",
    FieldType.DATE: "date",
    FieldType.TIME: "time",
    FieldType.EMAIL: "EmailStr",
    FieldType.URL: "HttpUrl",
    FieldType.UUID: "str",
    FieldType.JSON: "Dict[str, Any]",
    FieldType.ENUM: "str",
    FieldType.FILE: "str",
    FieldType.IMAGE: "str",
    FieldType.PASSWORD: "str",
    FieldType.PHONE: "str",
    FieldType.SLUG: "str",
    FieldType.IP_ADDRESS: "str",
    FieldType.BINARY: "bytes",
}

# Simple string → FieldType adapter for backward compatibility
SIMPLE_TYPE_ADAPTER: Dict[str, FieldType] = {
    "str": FieldType.STRING,
    "string": FieldType.STRING,
    "text": FieldType.TEXT,
    "int": FieldType.INTEGER,
    "integer": FieldType.INTEGER,
    "float": FieldType.FLOAT,
    "decimal": FieldType.DECIMAL,
    "bool": FieldType.BOOLEAN,
    "boolean": FieldType.BOOLEAN,
    "datetime": FieldType.DATETIME,
    "date": FieldType.DATE,
    "time": FieldType.TIME,
    "email": FieldType.EMAIL,
    "url": FieldType.URL,
    "uuid": FieldType.UUID,
    "json": FieldType.JSON,
    "dict": FieldType.JSON,
    "enum": FieldType.ENUM,
    "file": FieldType.FILE,
    "image": FieldType.IMAGE,
    "password": FieldType.PASSWORD,
    "phone": FieldType.PHONE,
    "slug": FieldType.SLUG,
    "ip": FieldType.IP_ADDRESS,
    "ip_address": FieldType.IP_ADDRESS,
    "binary": FieldType.BINARY,
}

# Auto-generated field names (excluded from user fields)
AUTO_FIELD_NAMES: frozenset = frozenset({
    "id", "created_at", "updated_at", "deleted_at", "is_deleted",
})

# Field types suitable for text search
SEARCHABLE_FIELD_TYPES: frozenset = frozenset({
    FieldType.STRING, FieldType.TEXT, FieldType.EMAIL,
    FieldType.SLUG, FieldType.PHONE,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resolve_field_type(type_input: Any) -> FieldType:
    """
    Resolve any type input to a FieldType enum.
    
    Accepts:
        - FieldType enum directly → O(1) passthrough
        - String name (e.g. "str", "email") → O(1) dict lookup
        - Python type objects (str, int, etc.) → O(1) dict lookup
    
    Args:
        type_input: Field type as FieldType, string, or Python type.
    
    Returns:
        Resolved FieldType enum.
    
    Raises:
        ValueError: If type cannot be resolved.
    """
    if isinstance(type_input, FieldType):
        return type_input
    
    if isinstance(type_input, str):
        normalized = type_input.lower().strip()
        resolved = SIMPLE_TYPE_ADAPTER.get(normalized)
        if resolved is not None:
            return resolved
        try:
            return FieldType(normalized)
        except ValueError:
            pass
        raise ValueError(
            f"Unknown field type: '{type_input}'. "
            f"Valid types: {sorted(SIMPLE_TYPE_ADAPTER.keys())}"
        )
    
    python_type_map: Dict[type, FieldType] = {
        str: FieldType.STRING,
        int: FieldType.INTEGER,
        float: FieldType.FLOAT,
        bool: FieldType.BOOLEAN,
        bytes: FieldType.BINARY,
        dict: FieldType.JSON,
        list: FieldType.JSON,
    }
    resolved = python_type_map.get(type_input)
    if resolved is not None:
        return resolved
    
    raise ValueError(
        f"Cannot resolve field type from: {type_input!r} (type={type(type_input).__name__})"
    )


def pluralize(name: str) -> str:
    """
    Simple English pluralization for table names.
    Covers 95% of common model names.
    
    Args:
        name: Singular lowercase name.
    
    Returns:
        Pluralized name.
    """
    if not name:
        return name
    
    if name.endswith("y") and len(name) > 1 and name[-2:] not in ("ay", "ey", "oy", "uy"):
        return name[:-1] + "ies"
    elif name.endswith(("s", "sh", "ch", "x", "z")):
        return name + "es"
    return name + "s"


def to_snake_case(name: str) -> str:
    """
    Convert PascalCase/camelCase to snake_case.
    
    Args:
        name: Input string in any case.
    
    Returns:
        snake_case string.
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def to_pascal_case(name: str) -> str:
    """
    Convert snake_case/kebab-case to PascalCase.
    
    Args:
        name: Input string.
    
    Returns:
        PascalCase string.
    """
    if "_" in name:
        return "".join(w.capitalize() for w in name.split("_"))
    if "-" in name:
        return "".join(w.capitalize() for w in name.split("-"))
    return name[0].upper() + name[1:] if name else name


# ============================================================
# FIELD CONSTRAINT (Pydantic V2)
# ============================================================

class FieldConstraint(BaseModel):
    """
    Validation constraints for a database field.
    Applied at both Pydantic schema level and optionally at database level.
    """
    min_length: Optional[int] = Field(default=None, ge=0)
    max_length: Optional[int] = Field(default=None, ge=1)
    min_value: Optional[float] = Field(default=None)
    max_value: Optional[float] = Field(default=None)
    regex_pattern: Optional[str] = Field(default=None)
    allowed_values: Optional[List[str]] = Field(default=None)
    custom_validator: Optional[str] = Field(default=None)
    
    @model_validator(mode="after")
    def validate_length_range(self) -> "FieldConstraint":
        """Ensure min_length <= max_length if both are set."""
        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError(
                    f"min_length ({self.min_length}) cannot exceed "
                    f"max_length ({self.max_length})"
                )
        return self
    
    @model_validator(mode="after")
    def validate_value_range(self) -> "FieldConstraint":
        """Ensure min_value <= max_value if both are set."""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) cannot exceed "
                    f"max_value ({self.max_value})"
                )
        return self
    
    def has_constraints(self) -> bool:
        """Check if any constraint is actively set. O(1)."""
        return (
            self.min_length is not None
            or self.max_length is not None
            or self.min_value is not None
            or self.max_value is not None
            or self.regex_pattern is not None
            or self.allowed_values is not None
            or self.custom_validator is not None
        )
    
    def to_pydantic_field_args(self) -> Dict[str, Any]:
        """Convert constraints to Pydantic Field() keyword arguments."""
        args: Dict[str, Any] = {}
        if self.min_length is not None:
            args["min_length"] = self.min_length
        if self.max_length is not None:
            args["max_length"] = self.max_length
        if self.min_value is not None:
            args["ge"] = self.min_value
        if self.max_value is not None:
            args["le"] = self.max_value
        return args


# ============================================================
# MODEL FIELD (Pydantic V2)
# ============================================================

class ModelField(BaseModel):
    """
    Complete definition of a single database model field.
    
    Generates:
    - SQLAlchemy Column(...) definitions
    - Pydantic schema field definitions
    - Pydantic validators
    """
    
    # Identity
    name: str = Field(..., min_length=1, max_length=64)
    field_type: FieldType
    
    # Behavior
    required: bool = Field(default=True)
    unique: bool = Field(default=False)
    indexed: bool = Field(default=False)
    nullable: bool = Field(default=False)
    primary_key: bool = Field(default=False)
    auto_increment: bool = Field(default=False)
    
    # Values
    default: Optional[Any] = Field(default=None)
    server_default: Optional[str] = Field(default=None)
    max_length: int = Field(default=255, ge=1, le=65535)
    precision: int = Field(default=10, ge=1, le=65)
    scale: int = Field(default=2, ge=0, le=30)
    
    # Metadata
    description: Optional[str] = Field(default=None, max_length=500)
    enum_values: Optional[List[str]] = Field(default=None)
    constraint: Optional[FieldConstraint] = Field(default=None)
    
    # Foreign key reference
    foreign_key: Optional[str] = Field(default=None)
    
    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Ensure name is a valid Python identifier."""
        if not v.isidentifier():
            raise ValueError(f"'{v}' is not a valid Python identifier")
        return v
    
    @model_validator(mode="after")
    def validate_enum_has_values(self) -> "ModelField":
        """Ensure enum fields have values defined."""
        if self.field_type == FieldType.ENUM and not self.enum_values:
            logger.warning(
                "Enum field '%s' has no enum_values defined. "
                "This will generate a plain string column.",
                self.name,
            )
        return self
    
    # ---- SQLAlchemy Generation ----
    
    def to_sqlalchemy(self) -> str:
        """
        Generate SQLAlchemy Column(...) definition string.
        
        Returns:
            Complete Column definition.
        """
        sa_template = SQLALCHEMY_TYPE_MAP.get(self.field_type, "String(255)")
        col_type = sa_template.format(
            max_length=self.max_length,
            precision=self.precision,
            scale=self.scale,
        )
        
        parts: List[str] = [f"Column({col_type}"]
        
        if self.primary_key:
            parts.append("primary_key=True")
        if self.auto_increment:
            parts.append("autoincrement=True")
        if self.foreign_key:
            parts.append(f'ForeignKey("{self.foreign_key}")')
        if self.unique:
            parts.append("unique=True")
        if self.indexed:
            parts.append("index=True")
        if self.nullable:
            parts.append("nullable=True")
        elif not self.primary_key:
            parts.append("nullable=False")
        
        if self.default is not None:
            if isinstance(self.default, str):
                parts.append(f'default="{self.default}"')
            elif isinstance(self.default, bool):
                parts.append(f"default={self.default}")
            else:
                parts.append(f"default={self.default}")
        
        if self.server_default is not None:
            parts.append(f'server_default=text("{self.server_default}")')
        
        return ", ".join(parts) + ")"
    
    # ---- Pydantic Generation ----
    
    def to_pydantic(self) -> str:
        """Generate Pydantic field definition for create schema."""
        py_type = PYDANTIC_TYPE_MAP.get(self.field_type, "str")
        
        if not self.required or self.nullable:
            py_type = f"Optional[{py_type}]"
        
        if self.default is not None:
            if isinstance(self.default, str):
                return f'    {self.name}: {py_type} = "{self.default}"'
            else:
                return f"    {self.name}: {py_type} = {self.default}"
        elif not self.required:
            return f"    {self.name}: {py_type} = None"
        else:
            return f"    {self.name}: {py_type}"
    
    def to_pydantic_update(self) -> str:
        """Generate Pydantic field for update schema (all optional)."""
        py_type = PYDANTIC_TYPE_MAP.get(self.field_type, "str")
        return f"    {self.name}: Optional[{py_type}] = None"
    
    # ---- Validator Generation ----
    
    def generate_validators(self) -> List[str]:
        """Generate Pydantic v2 field validators for this field."""
        validators: List[str] = []
        
        if self.constraint and self.constraint.has_constraints():
            c = self.constraint
            lines: List[str] = [
                f'    @field_validator("{self.name}")',
                f"    @classmethod",
                f"    def validate_{self.name}(cls, v: Any) -> Any:",
                f"        if v is None:",
                f"            return v",
            ]
            
            if c.min_length is not None:
                lines.append(f"        if len(str(v)) < {c.min_length}:")
                lines.append(
                    f'            raise ValueError("{self.name} must be at least {c.min_length} characters")'
                )
            
            if c.max_length is not None:
                lines.append(f"        if len(str(v)) > {c.max_length}:")
                lines.append(
                    f'            raise ValueError("{self.name} must be at most {c.max_length} characters")'
                )
            
            if c.min_value is not None:
                lines.append(f"        if float(v) < {c.min_value}:")
                lines.append(
                    f'            raise ValueError("{self.name} must be >= {c.min_value}")'
                )
            
            if c.max_value is not None:
                lines.append(f"        if float(v) > {c.max_value}:")
                lines.append(
                    f'            raise ValueError("{self.name} must be <= {c.max_value}")'
                )
            
            if c.regex_pattern is not None:
                lines.append(f"        import re")
                lines.append(f'        if not re.match(r"{c.regex_pattern}", str(v)):')
                lines.append(f'            raise ValueError("{self.name} has invalid format")')
            
            if c.allowed_values is not None:
                lines.append(f"        if v not in {c.allowed_values}:")
                lines.append(f'            raise ValueError("{self.name} must be one of {c.allowed_values}")')
            
            lines.append(f"        return v")
            validators.append("\n".join(lines))
        
        # Auto-validators based on field type
        if self.field_type == FieldType.EMAIL:
            validators.append(
                f'    @field_validator("{self.name}")\n'
                f"    @classmethod\n"
                f"    def validate_{self.name}_email(cls, v: Any) -> Any:\n"
                f"        if v is not None and '@' not in str(v):\n"
                f"            raise ValueError('Invalid email address')\n"
                f"        return v.lower().strip() if isinstance(v, str) else v"
            )
        
        if self.field_type == FieldType.SLUG:
            validators.append(
                f'    @field_validator("{self.name}")\n'
                f"    @classmethod\n"
                f"    def validate_{self.name}_slug(cls, v: Any) -> Any:\n"
                f"        import re\n"
                f"        if v and not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', str(v)):\n"
                f"            raise ValueError('Invalid slug format')\n"
                f"        return v"
            )
        
        if self.field_type == FieldType.PHONE:
            validators.append(
                f'    @field_validator("{self.name}")\n'
                f"    @classmethod\n"
                f"    def validate_{self.name}_phone(cls, v: Any) -> Any:\n"
                f"        import re\n"
                f"        if v and not re.match(r'^\\+?[0-9\\s\\-()]+$', str(v)):\n"
                f"            raise ValueError('Invalid phone number format')\n"
                f"        return v"
            )
        
        return validators
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize field to dictionary."""
        data: Dict[str, Any] = {
            "name": self.name,
            "type": self.field_type.value,
            "required": self.required,
            "unique": self.unique,
            "indexed": self.indexed,
            "nullable": self.nullable,
            "max_length": self.max_length,
        }
        if self.default is not None:
            data["default"] = self.default
        if self.description:
            data["description"] = self.description
        if self.enum_values:
            data["enum_values"] = self.enum_values
        if self.foreign_key:
            data["foreign_key"] = self.foreign_key
        return data


# ============================================================
# RELATIONSHIP (Pydantic V2)
# ============================================================

class Relationship(BaseModel):
    """
    Database relationship between two models.
    
    CRITICAL: source_model field tracks the owning model to fix
    association table foreign key generation bug.
    """
    
    # Required
    name: str = Field(..., min_length=1, max_length=64)
    target_model: str = Field(..., min_length=1, max_length=64)
    relation_type: RelationType
    
    # Source tracking (CRITICAL FIX)
    source_model: Optional[str] = Field(default=None)
    
    # Configuration
    back_populates: Optional[str] = Field(default=None)
    foreign_key: Optional[str] = Field(default=None)
    on_delete: OnDelete = Field(default=OnDelete.CASCADE)
    lazy: str = Field(default="selectin")
    association_table: Optional[str] = Field(default=None)
    
    # ---- Internal Helpers ----
    
    def _target_snake(self) -> str:
        """Convert target model name to snake_case."""
        return to_snake_case(self.target_model)
    
    def _source_snake(self) -> str:
        """Convert source model name to snake_case."""
        if self.source_model:
            return to_snake_case(self.source_model)
        return "unknown_source"
    
    def _target_table(self) -> str:
        """Derive target database table name."""
        return pluralize(self._target_snake())
    
    def _source_table(self) -> str:
        """
        Derive source database table name.
        
        CRITICAL FIX: Now correctly derives from source_model
        instead of returning hardcoded placeholder.
        """
        if self.source_model:
            return pluralize(self._source_snake())
        
        logger.warning(
            "Relationship '%s' → '%s' has no source_model set. "
            "Cannot derive correct source table name.",
            self.name,
            self.target_model,
        )
        return "unknown_sources"
    
    # ---- Code Generation ----
    
    def to_sqlalchemy_foreign_key(self) -> Optional[str]:
        """Generate SQLAlchemy foreign key Column definition."""
        if self.relation_type not in (RelationType.MANY_TO_ONE, RelationType.ONE_TO_ONE):
            return None
        
        fk_name = self.foreign_key or f"{self._target_snake()}_id"
        target_table = self._target_table()
        on_del = self.on_delete.value
        
        return (
            f"    {fk_name} = Column(Integer, "
            f'ForeignKey("{target_table}.id", ondelete="{on_del}"), '
            f"nullable=True)"
        )
    
    def to_sqlalchemy_relationship(self) -> str:
        """Generate SQLAlchemy relationship() declaration."""
        parts: List[str] = [f'relationship("{self.target_model}"']
        
        if self.back_populates:
            parts.append(f'back_populates="{self.back_populates}"')
        
        if self.relation_type == RelationType.ONE_TO_MANY:
            parts.append(f'lazy="{self.lazy}"')
            if self.on_delete == OnDelete.CASCADE:
                parts.append('cascade="all, delete-orphan"')
        
        elif self.relation_type == RelationType.ONE_TO_ONE:
            parts.append("uselist=False")
        
        elif self.relation_type == RelationType.MANY_TO_MANY:
            table_name = self._get_association_table_name()
            parts.append(f"secondary={table_name}")
            parts.append(f'lazy="{self.lazy}"')
        
        return f"    {self.name} = " + ", ".join(parts) + ")"
    
    def _get_association_table_name(self) -> str:
        """Derive association table name for M2M relationships."""
        if self.association_table:
            return self.association_table
        
        source = self._source_snake()
        target = self._target_snake()
        
        names = sorted([pluralize(source), pluralize(target)])
        return f"{'_'.join(names)}"
    
    def generate_association_table(self) -> Optional[str]:
        """
        Generate SQLAlchemy Table() definition for M2M junction tables.
        
        CRITICAL FIX: Uses _source_table() which correctly derives
        from source_model.
        """
        if self.relation_type != RelationType.MANY_TO_MANY:
            return None
        
        table_name = self._get_association_table_name()
        source_table = self._source_table()
        target_table = self._target_table()
        on_del = self.on_delete.value
        
        source_col = f"{self._source_snake()}_id"
        target_col = f"{self._target_snake()}_id"
        
        logger.debug(
            "Generating M2M association table: %s (%s ↔ %s)",
            table_name, source_table, target_table,
        )
        
        lines: List[str] = [
            f"{table_name} = Table(",
            f'    "{table_name}",',
            f"    Base.metadata,",
            f'    Column("{source_col}", Integer, '
            f'ForeignKey("{source_table}.id", ondelete="{on_del}"), '
            f"primary_key=True),",
            f'    Column("{target_col}", Integer, '
            f'ForeignKey("{target_table}.id", ondelete="{on_del}"), '
            f"primary_key=True),",
            f")",
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize relationship to dictionary."""
        return {
            "name": self.name,
            "target_model": self.target_model,
            "relation_type": self.relation_type.value,
            "source_model": self.source_model,
            "back_populates": self.back_populates,
            "on_delete": self.on_delete.value,
            "lazy": self.lazy,
        }


# ============================================================
# MODEL INDEX
# ============================================================

@dataclass
class ModelIndex:
    """Database index definition."""
    name: str
    fields: List[str]
    index_type: IndexType = IndexType.NORMAL
    
    def to_sqlalchemy(self, table_name: str) -> str:
        """Generate SQLAlchemy index/constraint string."""
        cols = ", ".join(f"'{f}'" for f in self.fields)
        
        if self.index_type == IndexType.UNIQUE:
            return f"UniqueConstraint({cols}, name='{self.name}')"
        else:
            return f"Index('{self.name}', {cols})"


# ============================================================
# DATABASE MODEL (Main Model Definition)
# ============================================================

class DatabaseModel(BaseModel):
    """
    Complete database model definition.
    
    Holds all information needed to generate:
    - SQLAlchemy ORM model class
    - Pydantic request/response schemas
    - CRUD operation functions
    - Router endpoint handlers
    """
    
    # Identity
    name: str = Field(..., min_length=2, max_length=64)
    description: Optional[str] = Field(default=None, max_length=500)
    
    # Field collections
    fields: List[ModelField] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    indexes: List[ModelIndex] = Field(default_factory=list)
    
    # Options
    table_name: Optional[str] = Field(default=None)
    timestamps: bool = Field(default=True)
    soft_delete: bool = Field(default=False)
    is_auth_model: bool = Field(default=False)
    
    model_config = {"arbitrary_types_allowed": True}
    
    def model_post_init(self, __context: Any) -> None:
        """Auto-inject standard fields based on configuration."""
        existing_names: Set[str] = {f.name for f in self.fields}
        
        # Primary key
        if "id" not in existing_names:
            self.fields.insert(0, ModelField(
                name="id",
                field_type=FieldType.INTEGER,
                primary_key=True,
                auto_increment=True,
                required=False,
                description="Primary key",
            ))
            logger.debug("Auto-injected 'id' field into '%s'", self.name)
        
        # Timestamps
        if self.timestamps:
            if "created_at" not in existing_names:
                self.fields.append(ModelField(
                    name="created_at",
                    field_type=FieldType.DATETIME,
                    required=False,
                    nullable=True,
                    server_default="now()",
                    description="Record creation timestamp",
                ))
            if "updated_at" not in existing_names:
                self.fields.append(ModelField(
                    name="updated_at",
                    field_type=FieldType.DATETIME,
                    required=False,
                    nullable=True,
                    server_default="now()",
                    description="Record update timestamp",
                ))
        
        # Soft delete
        if self.soft_delete:
            if "deleted_at" not in existing_names:
                self.fields.append(ModelField(
                    name="deleted_at",
                    field_type=FieldType.DATETIME,
                    required=False,
                    nullable=True,
                    description="Soft delete timestamp",
                ))
            if "is_deleted" not in existing_names:
                self.fields.append(ModelField(
                    name="is_deleted",
                    field_type=FieldType.BOOLEAN,
                    required=False,
                    default=False,
                    description="Soft delete flag",
                ))
        
        # Auth model fields
        if self.is_auth_model:
            self._inject_auth_fields(existing_names)
        
        logger.debug(
            "Model '%s' initialized: %d fields, %d relationships",
            self.name, len(self.fields), len(self.relationships),
        )
    
    def _inject_auth_fields(self, existing_names: Set[str]) -> None:
        """Inject authentication-related fields."""
        auth_fields = [
            ModelField(
                name="email",
                field_type=FieldType.EMAIL,
                unique=True,
                indexed=True,
                description="User email address",
            ),
            ModelField(
                name="hashed_password",
                field_type=FieldType.PASSWORD,
                description="Bcrypt hashed password",
            ),
            ModelField(
                name="is_active",
                field_type=FieldType.BOOLEAN,
                default=True,
                description="Account active flag",
            ),
            ModelField(
                name="is_superuser",
                field_type=FieldType.BOOLEAN,
                default=False,
                description="Superuser privilege flag",
            ),
            ModelField(
                name="last_login",
                field_type=FieldType.DATETIME,
                nullable=True,
                required=False,
                description="Last login timestamp",
            ),
        ]
        for af in auth_fields:
            if af.name not in existing_names:
                self.fields.append(af)
                logger.debug("Auto-injected auth field '%s'", af.name)
    
    # ---- Computed Properties ----
    
    @property
    def get_table_name(self) -> str:
        """Derive database table name."""
        if self.table_name:
            return self.table_name
        return pluralize(to_snake_case(self.name))
    
    @property
    def user_fields(self) -> List[ModelField]:
        """Get user-defined fields (excludes auto-generated)."""
        return [f for f in self.fields if f.name not in AUTO_FIELD_NAMES]
    
    @property
    def required_fields(self) -> List[ModelField]:
        """Get required fields for creation."""
        return [f for f in self.user_fields if f.required and f.default is None]
    
    @property
    def searchable_fields(self) -> List[ModelField]:
        """Get fields suitable for text search."""
        return [f for f in self.user_fields if f.field_type in SEARCHABLE_FIELD_TYPES]
    
    @property
    def unique_fields(self) -> List[ModelField]:
        """Get fields with unique constraints."""
        return [f for f in self.fields if f.unique]
    
    @property
    def indexed_fields(self) -> List[ModelField]:
        """Get explicitly indexed fields."""
        return [f for f in self.fields if f.indexed]
    
    # ---- SQLAlchemy Generation ----
    
    def generate_sqlalchemy_model(self) -> str:
        """Generate complete SQLAlchemy ORM model class."""
        lines: List[str] = []
        
        lines.append(f"class {self.name}(Base):")
        if self.description:
            lines.append(f'    """{self.description}"""')
        lines.append(f'    __tablename__ = "{self.get_table_name}"')
        lines.append("")
        
        for f in self.fields:
            if f.description:
                lines.append(f"    # {f.description}")
            lines.append(f"    {f.name} = {f.to_sqlalchemy()}")
        lines.append("")
        
        for rel in self.relationships:
            fk = rel.to_sqlalchemy_foreign_key()
            if fk:
                lines.append(fk)
            lines.append(rel.to_sqlalchemy_relationship())
        
        if self.relationships:
            lines.append("")
        
        if self.indexes:
            idx_strs = [idx.to_sqlalchemy(self.get_table_name) for idx in self.indexes]
            lines.append("    __table_args__ = (")
            for idx_str in idx_strs:
                lines.append(f"        {idx_str},")
            lines.append("    )")
            lines.append("")
        
        repr_fields = self.user_fields[:3]
        if repr_fields:
            repr_parts = ", ".join(f"{f.name}={{self.{f.name}}}" for f in repr_fields)
            lines.append("    def __repr__(self) -> str:")
            lines.append(f'        return f"<{self.name}({repr_parts})>"')
        
        return "\n".join(lines)
    
    # ---- Pydantic Schema Generation ----
    
    def generate_pydantic_schemas(self) -> str:
        """Generate all Pydantic schemas."""
        lines: List[str] = []
        user_fields = self.user_fields
        hidden_fields: Set[str] = {"hashed_password"}
        
        # Base Schema
        lines.append(f"class {self.name}Base(BaseModel):")
        if self.description:
            lines.append(f'    """{self.description} — base schema."""')
        
        schema_fields = [f for f in user_fields if f.name not in hidden_fields]
        if schema_fields:
            for f in schema_fields:
                lines.append(f.to_pydantic())
        else:
            lines.append("    pass")
        lines.append("")
        
        # Validators
        all_validators: List[str] = []
        for f in schema_fields:
            all_validators.extend(f.generate_validators())
        if all_validators:
            for v in all_validators:
                lines.append(v)
            lines.append("")
        
        # Create Schema
        lines.append(f"class {self.name}Create({self.name}Base):")
        lines.append(f'    """Schema for creating a new {self.name}."""')
        if self.is_auth_model:
            lines.append("    password: str")
        else:
            lines.append("    pass")
        lines.append("")
        
        # Update Schema
        lines.append(f"class {self.name}Update(BaseModel):")
        lines.append(f'    """Schema for updating {self.name}."""')
        for f in schema_fields:
            lines.append(f.to_pydantic_update())
        if self.is_auth_model:
            lines.append("    password: Optional[str] = None")
        lines.append("")
        
        # InDB Schema
        lines.append(f"class {self.name}InDB({self.name}Base):")
        lines.append(f'    """Schema for {self.name} from database."""')
        lines.append("    id: int")
        if self.timestamps:
            lines.append("    created_at: Optional[datetime] = None")
            lines.append("    updated_at: Optional[datetime] = None")
        if self.soft_delete:
            lines.append("    deleted_at: Optional[datetime] = None")
            lines.append("    is_deleted: bool = False")
        lines.append("")
        lines.append("    class Config:")
        lines.append("        from_attributes = True")
        lines.append("")
        
        # Response Schema
        lines.append(f"class {self.name}Response({self.name}InDB):")
        lines.append(f'    """API response schema for {self.name}."""')
        lines.append("    pass")
        
        return "\n".join(lines)
    
    def generate_association_tables(self) -> str:
        """Generate all M2M association Table() definitions."""
        tables: List[str] = []
        for rel in self.relationships:
            table = rel.generate_association_table()
            if table:
                tables.append(table)
        return "\n\n".join(tables)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "table_name": self.get_table_name,
            "fields": [f.to_dict() for f in self.fields],
            "relationships": [r.to_dict() for r in self.relationships],
            "timestamps": self.timestamps,
            "soft_delete": self.soft_delete,
            "is_auth_model": self.is_auth_model,
        }


# ============================================================
# FLUENT MODEL BUILDER
# ============================================================

class ModelBuilder:
    """
    Fluent builder for constructing DatabaseModel instances.
    All relationship methods automatically set source_model.
    """
    
    def __init__(self, name: str, description: str = "") -> None:
        self._name: str = name
        self._description: str = description
        self._fields: List[ModelField] = []
        self._relationships: List[Relationship] = []
        self._indexes: List[ModelIndex] = []
        self._timestamps: bool = True
        self._soft_delete: bool = False
        self._is_auth_model: bool = False
        self._table_name: Optional[str] = None
        
        logger.debug("ModelBuilder created for '%s'", name)
    
    # ---- Generic Field ----
    
    def add_field(
        self,
        name: str,
        field_type: Union[FieldType, str],
        required: bool = True,
        unique: bool = False,
        indexed: bool = False,
        nullable: bool = False,
        default: Optional[Any] = None,
        max_length: int = 255,
        description: Optional[str] = None,
        enum_values: Optional[List[str]] = None,
        constraint: Optional[FieldConstraint] = None,
        foreign_key: Optional[str] = None,
    ) -> "ModelBuilder":
        """Add a generic field."""
        resolved_type = resolve_field_type(field_type)
        self._fields.append(ModelField(
            name=name,
            field_type=resolved_type,
            required=required,
            unique=unique,
            indexed=indexed,
            nullable=nullable,
            default=default,
            max_length=max_length,
            description=description,
            enum_values=enum_values,
            constraint=constraint,
            foreign_key=foreign_key,
        ))
        return self
    
    # ---- Typed Shortcuts ----
    
    def add_string(self, name: str, max_length: int = 255, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.STRING, max_length=max_length, **kwargs)
    
    def add_text(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.TEXT, **kwargs)
    
    def add_integer(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.INTEGER, **kwargs)
    
    def add_float(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.FLOAT, **kwargs)
    
    def add_boolean(self, name: str, default: bool = False, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.BOOLEAN, default=default, **kwargs)
    
    def add_datetime(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.DATETIME, **kwargs)
    
    def add_date(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.DATE, **kwargs)
    
    def add_email(self, name: str = "email", **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.EMAIL, max_length=320, **kwargs)
    
    def add_url(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.URL, max_length=2048, **kwargs)
    
    def add_json(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.JSON, **kwargs)
    
    def add_enum(self, name: str, values: List[str], default: Optional[str] = None, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.ENUM, enum_values=values, default=default, **kwargs)
    
    def add_slug(self, name: str = "slug", **kwargs: Any) -> "ModelBuilder":
        kwargs.setdefault("unique", True)
        kwargs.setdefault("indexed", True)
        return self.add_field(name, FieldType.SLUG, **kwargs)
    
    def add_file(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.FILE, max_length=512, **kwargs)
    
    def add_image(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.IMAGE, max_length=512, **kwargs)
    
    def add_password(self, name: str = "hashed_password", **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.PASSWORD, **kwargs)
    
    def add_phone(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.PHONE, max_length=20, **kwargs)
    
    def add_uuid(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.UUID, **kwargs)
    
    def add_decimal(self, name: str, **kwargs: Any) -> "ModelBuilder":
        return self.add_field(name, FieldType.DECIMAL, **kwargs)
    
    # ---- Relationships (all set source_model) ----
    
    def belongs_to(
        self,
        name: str,
        target: str,
        back_populates: Optional[str] = None,
        on_delete: OnDelete = OnDelete.CASCADE,
        foreign_key: Optional[str] = None,
    ) -> "ModelBuilder":
        """Add a many-to-one relationship."""
        self._relationships.append(Relationship(
            name=name,
            target_model=target,
            relation_type=RelationType.MANY_TO_ONE,
            source_model=self._name,
            back_populates=back_populates,
            on_delete=on_delete,
            foreign_key=foreign_key,
        ))
        return self
    
    def has_many(
        self,
        name: str,
        target: str,
        back_populates: Optional[str] = None,
        on_delete: OnDelete = OnDelete.CASCADE,
    ) -> "ModelBuilder":
        """Add a one-to-many relationship."""
        self._relationships.append(Relationship(
            name=name,
            target_model=target,
            relation_type=RelationType.ONE_TO_MANY,
            source_model=self._name,
            back_populates=back_populates,
            on_delete=on_delete,
        ))
        return self
    
    def has_one(
        self,
        name: str,
        target: str,
        back_populates: Optional[str] = None,
        on_delete: OnDelete = OnDelete.CASCADE,
    ) -> "ModelBuilder":
        """Add a one-to-one relationship."""
        self._relationships.append(Relationship(
            name=name,
            target_model=target,
            relation_type=RelationType.ONE_TO_ONE,
            source_model=self._name,
            back_populates=back_populates,
            on_delete=on_delete,
        ))
        return self
    
    def many_to_many(
        self,
        name: str,
        target: str,
        back_populates: Optional[str] = None,
        on_delete: OnDelete = OnDelete.CASCADE,
        association_table: Optional[str] = None,
    ) -> "ModelBuilder":
        """Add a many-to-many relationship."""
        self._relationships.append(Relationship(
            name=name,
            target_model=target,
            relation_type=RelationType.MANY_TO_MANY,
            source_model=self._name,
            back_populates=back_populates,
            on_delete=on_delete,
            association_table=association_table,
        ))
        return self
    
    # ---- Configuration ----
    
    def with_timestamps(self, enabled: bool = True) -> "ModelBuilder":
        self._timestamps = enabled
        return self
    
    def with_soft_delete(self, enabled: bool = True) -> "ModelBuilder":
        self._soft_delete = enabled
        return self
    
    def with_table_name(self, table_name: str) -> "ModelBuilder":
        self._table_name = table_name
        return self
    
    def as_auth_model(self) -> "ModelBuilder":
        self._is_auth_model = True
        return self
    
    # ---- Indexes ----
    
    def add_index(
        self,
        fields: List[str],
        index_type: IndexType = IndexType.NORMAL,
        name: Optional[str] = None,
    ) -> "ModelBuilder":
        idx_name = name or f"idx_{self._name.lower()}_{'_'.join(fields)}"
        self._indexes.append(ModelIndex(
            name=idx_name,
            fields=fields,
            index_type=index_type,
        ))
        return self
    
    def add_unique_index(self, fields: List[str], name: Optional[str] = None) -> "ModelBuilder":
        return self.add_index(fields, IndexType.UNIQUE, name)
    
    # ---- Build ----
    
    def build(self) -> DatabaseModel:
        """Build the final DatabaseModel."""
        model = DatabaseModel(
            name=self._name,
            description=self._description,
            fields=self._fields,
            relationships=self._relationships,
            indexes=self._indexes,
            table_name=self._table_name,
            timestamps=self._timestamps,
            soft_delete=self._soft_delete,
            is_auth_model=self._is_auth_model,
        )
        
        logger.info(
            "Built model '%s': %d fields, %d relationships",
            model.name, len(model.fields), len(model.relationships),
        )
        
        return model


# ============================================================
# PRESET MODEL FACTORIES
# ============================================================

def create_user_model() -> DatabaseModel:
    """Create standard User model with auth."""
    return (
        ModelBuilder("User", "Application user")
        .add_string("username", max_length=50, unique=True, indexed=True)
        .add_string("full_name", max_length=100, required=False, nullable=True)
        .as_auth_model()
        .with_timestamps()
        .with_soft_delete()
        .build()
    )


def create_post_model(author_model: str = "User") -> DatabaseModel:
    """Create standard Post model."""
    return (
        ModelBuilder("Post", "Blog post")
        .add_string("title", max_length=200)
        .add_slug("slug")
        .add_text("content")
        .add_text("excerpt", required=False, nullable=True)
        .add_enum("status", ["draft", "published", "archived"], default="draft")
        .add_integer("view_count", default=0, required=False)
        .add_datetime("published_at", required=False, nullable=True)
        .belongs_to("author", author_model, back_populates="posts")
        .with_timestamps()
        .build()
    )


def create_tag_model(taggable_model: str = "Post") -> DatabaseModel:
    """Create Tag model with M2M relationship."""
    return (
        ModelBuilder("Tag", "Content tag")
        .add_string("name", max_length=50, unique=True)
        .add_slug("slug")
        .add_string("color", max_length=7, required=False, nullable=True)
        .many_to_many("posts", taggable_model, back_populates="tags")
        .with_timestamps()
        .build()
    )
