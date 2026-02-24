"""
APIGen - Database Model System
Advanced model definition with relationships, constraints, and auto-migrations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class FieldType(Enum):
    """Supported database field types."""
    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
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
    

class RelationType(Enum):
    """Database relationship types."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class IndexType(Enum):
    """Database index types."""
    NORMAL = "index"
    UNIQUE = "unique"
    COMPOSITE = "composite"
    FULLTEXT = "fulltext"


class OnDelete(Enum):
    """Foreign key on delete actions."""
    CASCADE = "CASCADE"
    SET_NULL = "SET NULL"
    RESTRICT = "RESTRICT"
    NO_ACTION = "NO ACTION"
    SET_DEFAULT = "SET DEFAULT"


# ============================================================
# FIELD DEFINITION
# ============================================================

@dataclass
class FieldConstraint:
    """Constraints for a database field."""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    regex_pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    custom_validator: Optional[str] = None


@dataclass
class ModelField:
    """Definition of a single model field."""
    name: str
    field_type: FieldType
    required: bool = True
    unique: bool = False
    indexed: bool = False
    nullable: bool = False
    default: Optional[Any] = None
    primary_key: bool = False
    auto_increment: bool = False
    server_default: Optional[str] = None
    description: Optional[str] = None
    max_length: int = 255
    constraint: Optional[FieldConstraint] = None
    
    # For enum fields
    enum_values: Optional[List[str]] = None
    
    def to_sqlalchemy(self) -> str:
        """Generate SQLAlchemy column definition."""
        type_map = {
            FieldType.STRING: f"String({self.max_length})",
            FieldType.TEXT: "Text",
            FieldType.INTEGER: "Integer",
            FieldType.FLOAT: "Float",
            FieldType.BOOLEAN: "Boolean",
            FieldType.DATETIME: "DateTime(timezone=True)",
            FieldType.DATE: "Date",
            FieldType.EMAIL: "String(320)",
            FieldType.URL: "String(2048)",
            FieldType.UUID: "String(36)",
            FieldType.JSON: "JSON",
            FieldType.ENUM: f"String(50)",
            FieldType.FILE: "String(512)",
            FieldType.IMAGE: "String(512)",
            FieldType.PASSWORD: "String(128)",
            FieldType.PHONE: "String(20)",
            FieldType.SLUG: f"String({self.max_length})",
            FieldType.IP_ADDRESS: "String(45)",
        }
        
        col_type = type_map.get(self.field_type, "String(255)")
        
        parts = [f"Column({col_type}"]
        
        if self.primary_key:
            parts.append("primary_key=True")
        if self.auto_increment:
            parts.append("autoincrement=True")
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
        if self.server_default:
            parts.append(f'server_default=text("{self.server_default}")')
        
        return ", ".join(parts) + ")"
    
    def to_pydantic(self) -> str:
        """Generate Pydantic field definition."""
        type_map = {
            FieldType.STRING: "str",
            FieldType.TEXT: "str",
            FieldType.INTEGER: "int",
            FieldType.FLOAT: "float",
            FieldType.BOOLEAN: "bool",
            FieldType.DATETIME: "datetime",
            FieldType.DATE: "date",
            FieldType.EMAIL: "EmailStr",
            FieldType.URL: "HttpUrl",
            FieldType.UUID: "str",
            FieldType.JSON: "dict",
            FieldType.ENUM: "str",
            FieldType.FILE: "str",
            FieldType.IMAGE: "str",
            FieldType.PASSWORD: "str",
            FieldType.PHONE: "str",
            FieldType.SLUG: "str",
            FieldType.IP_ADDRESS: "str",
        }
        
        py_type = type_map.get(self.field_type, "str")
        
        if not self.required or self.nullable:
            py_type = f"Optional[{py_type}]"
        
        if self.default is not None:
            if isinstance(self.default, str):
                return f'{self.name}: {py_type} = "{self.default}"'
            else:
                return f"{self.name}: {py_type} = {self.default}"
        elif not self.required:
            return f"{self.name}: {py_type} = None"
        else:
            return f"{self.name}: {py_type}"
    
    def get_validators(self) -> List[str]:
        """Generate Pydantic validators for this field."""
        validators = []
        
        if self.constraint:
            c = self.constraint
            
            if c.min_length is not None or c.max_length is not None:
                validators.append(
                    f"    @validator('{self.name}')\n"
                    f"    def validate_{self.name}_length(cls, v):\n"
                    f"        if v is not None:\n"
                )
                if c.min_length is not None:
                    validators.append(
                        f"            if len(str(v)) < {c.min_length}:\n"
                        f"                raise ValueError('{self.name} must be at least {c.min_length} characters')\n"
                    )
                if c.max_length is not None:
                    validators.append(
                        f"            if len(str(v)) > {c.max_length}:\n"
                        f"                raise ValueError('{self.name} must be at most {c.max_length} characters')\n"
                    )
                validators.append("        return v\n")
            
            if c.regex_pattern:
                validators.append(
                    f"    @validator('{self.name}')\n"
                    f"    def validate_{self.name}_pattern(cls, v):\n"
                    f"        import re\n"
                    f"        if v and not re.match(r'{c.regex_pattern}', str(v)):\n"
                    f"            raise ValueError('{self.name} has invalid format')\n"
                    f"        return v\n"
                )
            
            if c.allowed_values:
                vals = ", ".join(f'"{v}"' for v in c.allowed_values)
                validators.append(
                    f"    @validator('{self.name}')\n"
                    f"    def validate_{self.name}_values(cls, v):\n"
                    f"        allowed = [{vals}]\n"
                    f"        if v not in allowed:\n"
                    f"            raise ValueError(f'{self.name} must be one of {{allowed}}')\n"
                    f"        return v\n"
                )
        
        if self.field_type == FieldType.EMAIL:
            validators.append(
                f"    @validator('{self.name}')\n"
                f"    def validate_{self.name}_email(cls, v):\n"
                f"        if v and '@' not in v:\n"
                f"            raise ValueError('Invalid email address')\n"
                f"        return v.lower().strip() if v else v\n"
            )
        
        if self.field_type == FieldType.SLUG:
            validators.append(
                f"    @validator('{self.name}')\n"
                f"    def validate_{self.name}_slug(cls, v):\n"
                f"        import re\n"
                f"        if v and not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', v):\n"
                f"            raise ValueError('Invalid slug format')\n"
                f"        return v\n"
            )
        
        return validators


# ============================================================
# RELATIONSHIP DEFINITION
# ============================================================

@dataclass
class Relationship:
    """Definition of a model relationship."""
    name: str
    target_model: str
    relation_type: RelationType
    back_populates: Optional[str] = None
    foreign_key: Optional[str] = None
    on_delete: OnDelete = OnDelete.CASCADE
    lazy: str = "selectin"
    
    # For many-to-many
    association_table: Optional[str] = None
    
    def to_sqlalchemy_foreign_key(self) -> Optional[str]:
        """Generate foreign key column if needed."""
        if self.relation_type in (RelationType.MANY_TO_ONE, RelationType.ONE_TO_ONE):
            fk_name = self.foreign_key or f"{self._target_snake()}_id"
            return (
                f"    {fk_name} = Column(Integer, "
                f'ForeignKey("{self._target_table()}.id", '
                f'ondelete="{self.on_delete.value}"), '
                f"nullable=False)"
            )
        return None
    
    def to_sqlalchemy_relationship(self) -> str:
        """Generate SQLAlchemy relationship definition."""
        parts = [f'relationship("{self.target_model}"']
        
        if self.back_populates:
            parts.append(f'back_populates="{self.back_populates}"')
        
        if self.relation_type == RelationType.ONE_TO_MANY:
            parts.append(f'lazy="{self.lazy}"')
            if self.on_delete == OnDelete.CASCADE:
                parts.append('cascade="all, delete-orphan"')
        
        if self.relation_type == RelationType.ONE_TO_ONE:
            parts.append("uselist=False")
        
        if self.relation_type == RelationType.MANY_TO_MANY:
            table = self.association_table or f"{self._target_snake()}_association"
            parts.append(f"secondary={table}")
            parts.append(f'lazy="{self.lazy}"')
        
        return f"    {self.name} = " + ", ".join(parts) + ")"
    
    def generate_association_table(self) -> Optional[str]:
        """Generate many-to-many association table."""
        if self.relation_type != RelationType.MANY_TO_MANY:
            return None
        
        table_name = self.association_table or f"{self._target_snake()}_association"
        
        return (
            f"{table_name} = Table(\n"
            f"    '{table_name}',\n"
            f"    Base.metadata,\n"
            f"    Column('left_id', Integer, ForeignKey('{self._source_table()}.id'), primary_key=True),\n"
            f"    Column('right_id', Integer, ForeignKey('{self._target_table()}.id'), primary_key=True),\n"
            f")\n"
        )
    
    def _target_snake(self) -> str:
        """Convert target model to snake_case."""
        import re
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', self.target_model)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _target_table(self) -> str:
        """Get target table name (pluralized snake_case)."""
        snake = self._target_snake()
        if snake.endswith('y'):
            return snake[:-1] + 'ies'
        elif snake.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return snake + 'es'
        return snake + 's'
    
    def _source_table(self) -> str:
        """Placeholder for source table name."""
        return "source_table"


# ============================================================
# INDEX DEFINITION
# ============================================================

@dataclass
class ModelIndex:
    """Database index definition."""
    name: str
    fields: List[str]
    index_type: IndexType = IndexType.NORMAL
    
    def to_sqlalchemy(self, table_name: str) -> str:
        """Generate SQLAlchemy index."""
        cols = ", ".join(f"'{f}'" for f in self.fields)
        
        if self.index_type == IndexType.UNIQUE:
            return f"UniqueConstraint({cols}, name='{self.name}')"
        else:
            return f"Index('{self.name}', {cols})"


# ============================================================
# COMPLETE MODEL
# ============================================================

@dataclass
class DatabaseModel:
    """Complete database model definition."""
    name: str
    fields: List[ModelField] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    indexes: List[ModelIndex] = field(default_factory=list)
    
    # Options
    table_name: Optional[str] = None
    timestamps: bool = True
    soft_delete: bool = False
    description: Optional[str] = None
    
    # Auth model flags
    is_auth_model: bool = False
    
    def __post_init__(self):
        """Add default fields after initialization."""
        has_id = any(f.name == 'id' for f in self.fields)
        if not has_id:
            id_field = ModelField(
                name="id",
                field_type=FieldType.INTEGER,
                primary_key=True,
                auto_increment=True,
                required=False,
                description="Primary key"
            )
            self.fields.insert(0, id_field)
        
        if self.timestamps:
            has_created = any(f.name == 'created_at' for f in self.fields)
            has_updated = any(f.name == 'updated_at' for f in self.fields)
            
            if not has_created:
                self.fields.append(ModelField(
                    name="created_at",
                    field_type=FieldType.DATETIME,
                    required=False,
                    nullable=True,
                    server_default="now()",
                    description="Record creation timestamp"
                ))
            
            if not has_updated:
                self.fields.append(ModelField(
                    name="updated_at",
                    field_type=FieldType.DATETIME,
                    required=False,
                    nullable=True,
                    server_default="now()",
                    description="Record update timestamp"
                ))
        
        if self.soft_delete:
            has_deleted = any(f.name == 'deleted_at' for f in self.fields)
            if not has_deleted:
                self.fields.append(ModelField(
                    name="deleted_at",
                    field_type=FieldType.DATETIME,
                    required=False,
                    nullable=True,
                    description="Soft delete timestamp"
                ))
                self.fields.append(ModelField(
                    name="is_deleted",
                    field_type=FieldType.BOOLEAN,
                    required=False,
                    default=False,
                    description="Soft delete flag"
                ))
        
        if self.is_auth_model:
            self._add_auth_fields()
    
    def _add_auth_fields(self):
        """Add authentication-related fields."""
        auth_fields = [
            ModelField(name="email", field_type=FieldType.EMAIL, unique=True, indexed=True),
            ModelField(name="hashed_password", field_type=FieldType.PASSWORD),
            ModelField(name="is_active", field_type=FieldType.BOOLEAN, default=True),
            ModelField(name="is_superuser", field_type=FieldType.BOOLEAN, default=False),
            ModelField(name="last_login", field_type=FieldType.DATETIME, nullable=True, required=False),
        ]
        
        existing_names = {f.name for f in self.fields}
        for af in auth_fields:
            if af.name not in existing_names:
                self.fields.append(af)
    
    @property
    def get_table_name(self) -> str:
        """Get the database table name."""
        if self.table_name:
            return self.table_name
        
        import re
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', self.name)
        snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
        if snake.endswith('y') and snake[-2:] not in ('ay', 'ey', 'oy', 'uy'):
            return snake[:-1] + 'ies'
        elif snake.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return snake + 'es'
        return snake + 's'
    
    @property
    def user_fields(self) -> List[ModelField]:
        """Get only user-defined fields (no id, timestamps, soft_delete)."""
        auto_fields = {'id', 'created_at', 'updated_at', 'deleted_at', 'is_deleted'}
        return [f for f in self.fields if f.name not in auto_fields]
    
    @property
    def required_fields(self) -> List[ModelField]:
        """Get required fields for creation."""
        return [f for f in self.user_fields if f.required and f.default is None]
    
    @property
    def searchable_fields(self) -> List[ModelField]:
        """Get fields that can be searched/filtered."""
        searchable_types = {
            FieldType.STRING, FieldType.TEXT, FieldType.EMAIL,
            FieldType.SLUG, FieldType.PHONE
        }
        return [f for f in self.user_fields if f.field_type in searchable_types]
    
    def generate_sqlalchemy_model(self) -> str:
        """Generate complete SQLAlchemy model code."""
        lines = []
        
        lines.append(f"class {self.name}(Base):")
        if self.description:
            lines.append(f'    """{self.description}"""')
        lines.append(f'    __tablename__ = "{self.get_table_name}"')
        lines.append("")
        
        # Columns
        for f in self.fields:
            if f.description:
                lines.append(f"    # {f.description}")
            lines.append(f"    {f.name} = {f.to_sqlalchemy()}")
        
        lines.append("")
        
        # Relationships
        for rel in self.relationships:
            fk = rel.to_sqlalchemy_foreign_key()
            if fk:
                lines.append(fk)
            lines.append(rel.to_sqlalchemy_relationship())
        
        if self.relationships:
            lines.append("")
        
        # Table args for indexes
        if self.indexes:
            idx_strs = [idx.to_sqlalchemy(self.get_table_name) for idx in self.indexes]
            lines.append(f"    __table_args__ = (")
            for idx_str in idx_strs:
                lines.append(f"        {idx_str},")
            lines.append(f"    )")
            lines.append("")
        
        # __repr__
        repr_fields = [f for f in self.user_fields[:3]]
        if repr_fields:
            repr_parts = ", ".join(f'{f.name}={{self.{f.name}}}' for f in repr_fields)
            lines.append(f"    def __repr__(self):")
            lines.append(f'        return f"<{self.name}({repr_parts})>"')
        
        return "\n".join(lines)
    
    def generate_pydantic_schemas(self) -> str:
        """Generate Pydantic schemas (Base, Create, Update, Response)."""
        lines = []
        user_fields = self.user_fields
        
        # Base Schema
        lines.append(f"class {self.name}Base(BaseModel):")
        if self.description:
            lines.append(f'    """{self.description} - Base schema."""')
        for f in user_fields:
            lines.append(f"    {f.to_pydantic()}")
        lines.append("")
        
        # Validators
        all_validators = []
        for f in user_fields:
            all_validators.extend(f.get_validators())
        if all_validators:
            for v in all_validators:
                lines.append(v)
            lines.append("")
        
        # Create Schema
        lines.append(f"class {self.name}Create({self.name}Base):")
        lines.append(f'    """Schema for creating {self.name}."""')
        lines.append("    pass")
        lines.append("")
        
        # Update Schema (all fields optional)
        lines.append(f"class {self.name}Update(BaseModel):")
        lines.append(f'    """Schema for updating {self.name}."""')
        for f in user_fields:
            py_type = f.to_pydantic().split(":")[1].strip() if ":" in f.to_pydantic() else "str"
            base_type = py_type.split("=")[0].strip()
            if "Optional" not in base_type:
                base_type = f"Optional[{base_type}]"
            lines.append(f"    {f.name}: {base_type} = None")
        lines.append("")
        
        # Response Schema
        lines.append(f"class {self.name}Response({self.name}Base):")
        lines.append(f'    """Schema for {self.name} API response."""')
        lines.append("    id: int")
        if self.timestamps:
            lines.append("    created_at: Optional[datetime] = None")
            lines.append("    updated_at: Optional[datetime] = None")
        lines.append("")
        lines.append("    class Config:")
        lines.append("        from_attributes = True")
        lines.append("")
        
        # List Response
        lines.append(f"class {self.name}ListResponse(BaseModel):")
        lines.append(f'    """Paginated list response for {self.name}."""')
        lines.append(f"    items: List[{self.name}Response]")
        lines.append("    total: int")
        lines.append("    page: int = 1")
        lines.append("    per_page: int = 20")
        lines.append("    pages: int = 1")
        
        return "\n".join(lines)
    
    def generate_crud(self) -> str:
        """Generate CRUD operations."""
        snake = self.get_table_name.rstrip('s')
        lines = []
        
        lines.append(f"class {self.name}CRUD:")
        lines.append(f'    """CRUD operations for {self.name}."""')
        lines.append("")
        
        # Get by ID
        lines.append(f"    @staticmethod")
        lines.append(f"    async def get(db: AsyncSession, {snake}_id: int) -> Optional[{self.name}]:")
        lines.append(f"        result = await db.execute(select({self.name}).filter({self.name}.id == {snake}_id))")
        lines.append(f"        return result.scalar_one_or_none()")
        lines.append("")
        
        # Get all with pagination
        lines.append(f"    @staticmethod")
        lines.append(f"    async def get_all(db: AsyncSession, skip: int = 0, limit: int = 20) -> List[{self.name}]:")
        if self.soft_delete:
            lines.append(f"        result = await db.execute(")
            lines.append(f"            select({self.name}).filter({self.name}.is_deleted == False).offset(skip).limit(limit)")
            lines.append(f"        )")
        else:
            lines.append(f"        result = await db.execute(select({self.name}).offset(skip).limit(limit))")
        lines.append(f"        return result.scalars().all()")
        lines.append("")
        
        # Count
        lines.append(f"    @staticmethod")
        lines.append(f"    async def count(db: AsyncSession) -> int:")
        lines.append(f"        result = await db.execute(select(func.count({self.name}.id)))")
        lines.append(f"        return result.scalar()")
        lines.append("")
        
        # Create
        lines.append(f"    @staticmethod")
        lines.append(f"    async def create(db: AsyncSession, data: {self.name}Create) -> {self.name}:")
        lines.append(f"        db_obj = {self.name}(**data.model_dump())")
        lines.append(f"        db.add(db_obj)")
        lines.append(f"        await db.commit()")
        lines.append(f"        await db.refresh(db_obj)")
        lines.append(f"        return db_obj")
        lines.append("")
        
        # Update
        lines.append(f"    @staticmethod")
        lines.append(f"    async def update(db: AsyncSession, {snake}_id: int, data: {self.name}Update) -> Optional[{self.name}]:")
        lines.append(f"        db_obj = await {self.name}CRUD.get(db, {snake}_id)")
        lines.append(f"        if not db_obj:")
        lines.append(f"            return None")
        lines.append(f"        update_data = data.model_dump(exclude_unset=True)")
        lines.append(f"        for field, value in update_data.items():")
        lines.append(f"            setattr(db_obj, field, value)")
        lines.append(f"        await db.commit()")
        lines.append(f"        await db.refresh(db_obj)")
        lines.append(f"        return db_obj")
        lines.append("")
        
        # Delete
        lines.append(f"    @staticmethod")
        lines.append(f"    async def delete(db: AsyncSession, {snake}_id: int) -> bool:")
        lines.append(f"        db_obj = await {self.name}CRUD.get(db, {snake}_id)")
        lines.append(f"        if not db_obj:")
        lines.append(f"            return False")
        if self.soft_delete:
            lines.append(f"        db_obj.is_deleted = True")
            lines.append(f"        db_obj.deleted_at = func.now()")
            lines.append(f"        await db.commit()")
        else:
            lines.append(f"        await db.delete(db_obj)")
            lines.append(f"        await db.commit()")
        lines.append(f"        return True")
        lines.append("")
        
        # Search (if searchable fields exist)
        search_fields = self.searchable_fields
        if search_fields:
            lines.append(f"    @staticmethod")
            lines.append(f"    async def search(db: AsyncSession, query: str, limit: int = 20) -> List[{self.name}]:")
            conditions = " | ".join(
                f"{self.name}.{f.name}.ilike(f'%{{query}}%')" for f in search_fields
            )
            lines.append(f"        result = await db.execute(")
            lines.append(f"            select({self.name}).filter({conditions}).limit(limit)")
            lines.append(f"        )")
            lines.append(f"        return result.scalars().all()")
        
        return "\n".join(lines)


# ============================================================
# MODEL BUILDER (FLUENT API)
# ============================================================

class ModelBuilder:
    """Fluent API for building database models.
    
    Usage:
        model = (ModelBuilder("User")
            .add_string("username", unique=True, max_length=50)
            .add_email("email", unique=True)
            .add_password("password")
            .add_boolean("is_active", default=True)
            .with_timestamps()
            .with_soft_delete()
            .build())
    """
    
    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
        self._fields: List[ModelField] = []
        self._relationships: List[Relationship] = []
        self._indexes: List[ModelIndex] = []
        self._timestamps = True
        self._soft_delete = False
        self._is_auth = False
        self._table_name: Optional[str] = None
    
    def add_field(self, name: str, field_type: FieldType, **kwargs) -> 'ModelBuilder':
        """Add a generic field."""
        self._fields.append(ModelField(name=name, field_type=field_type, **kwargs))
        return self
    
    def add_string(self, name: str, max_length: int = 255, **kwargs) -> 'ModelBuilder':
        """Add a string field."""
        return self.add_field(name, FieldType.STRING, max_length=max_length, **kwargs)
    
    def add_text(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add a text field."""
        return self.add_field(name, FieldType.TEXT, **kwargs)
    
    def add_integer(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add an integer field."""
        return self.add_field(name, FieldType.INTEGER, **kwargs)
    
    def add_float(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add a float field."""
        return self.add_field(name, FieldType.FLOAT, **kwargs)
    
    def add_boolean(self, name: str, default: bool = False, **kwargs) -> 'ModelBuilder':
        """Add a boolean field."""
        return self.add_field(name, FieldType.BOOLEAN, default=default, **kwargs)
    
    def add_datetime(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add a datetime field."""
        return self.add_field(name, FieldType.DATETIME, **kwargs)
    
    def add_email(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add an email field."""
        return self.add_field(name, FieldType.EMAIL, **kwargs)
    
    def add_url(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add a URL field."""
        return self.add_field(name, FieldType.URL, **kwargs)
    
    def add_password(self, name: str = "password", **kwargs) -> 'ModelBuilder':
        """Add a password field."""
        return self.add_field(name, FieldType.PASSWORD, **kwargs)
    
    def add_slug(self, name: str = "slug", **kwargs) -> 'ModelBuilder':
        """Add a slug field."""
        return self.add_field(name, FieldType.SLUG, unique=True, indexed=True, **kwargs)
    
    def add_enum(self, name: str, values: List[str], **kwargs) -> 'ModelBuilder':
        """Add an enum field."""
        self._fields.append(ModelField(
            name=name, field_type=FieldType.ENUM,
            enum_values=values, **kwargs
        ))
        return self
    
    def add_json(self, name: str, **kwargs) -> 'ModelBuilder':
        """Add a JSON field."""
        return self.add_field(name, FieldType.JSON, **kwargs)
    
    def has_one(self, name: str, target: str, **kwargs) -> 'ModelBuilder':
        """Add a one-to-one relationship."""
        self._relationships.append(Relationship(
            name=name, target_model=target,
            relation_type=RelationType.ONE_TO_ONE, **kwargs
        ))
        return self
    
    def has_many(self, name: str, target: str, **kwargs) -> 'ModelBuilder':
        """Add a one-to-many relationship."""
        self._relationships.append(Relationship(
            name=name, target_model=target,
            relation_type=RelationType.ONE_TO_MANY, **kwargs
        ))
        return self
    
    def belongs_to(self, name: str, target: str, **kwargs) -> 'ModelBuilder':
        """Add a many-to-one relationship."""
        self._relationships.append(Relationship(
            name=name, target_model=target,
            relation_type=RelationType.MANY_TO_ONE, **kwargs
        ))
        return self
    
    def many_to_many(self, name: str, target: str, **kwargs) -> 'ModelBuilder':
        """Add a many-to-many relationship."""
        self._relationships.append(Relationship(
            name=name, target_model=target,
            relation_type=RelationType.MANY_TO_MANY, **kwargs
        ))
        return self
    
    def add_index(self, name: str, fields: List[str],
                  index_type: IndexType = IndexType.NORMAL) -> 'ModelBuilder':
        """Add a database index."""
        self._indexes.append(ModelIndex(name=name, fields=fields, index_type=index_type))
        return self
    
    def with_timestamps(self, enabled: bool = True) -> 'ModelBuilder':
        """Enable/disable automatic timestamps."""
        self._timestamps = enabled
        return self
    
    def with_soft_delete(self, enabled: bool = True) -> 'ModelBuilder':
        """Enable/disable soft delete."""
        self._soft_delete = enabled
        return self
    
    def as_auth_model(self) -> 'ModelBuilder':
        """Mark as authentication model (adds email, password, etc.)."""
        self._is_auth = True
        return self
    
    def set_table_name(self, name: str) -> 'ModelBuilder':
        """Set custom table name."""
        self._table_name = name
        return self
    
    def build(self) -> DatabaseModel:
        """Build the final DatabaseModel."""
        return DatabaseModel(
            name=self._name,
            fields=self._fields,
            relationships=self._relationships,
            indexes=self._indexes,
            table_name=self._table_name,
            timestamps=self._timestamps,
            soft_delete=self._soft_delete,
            description=self._description,
            is_auth_model=self._is_auth,
        )


# ============================================================
# QUICK MODEL FACTORIES
# ============================================================

def create_user_model(extra_fields: Optional[List[ModelField]] = None) -> DatabaseModel:
    """Create a standard User model."""
    builder = (ModelBuilder("User", "Application user model")
        .add_string("username", max_length=50, unique=True, indexed=True)
        .add_string("full_name", max_length=100, required=False, nullable=True)
        .as_auth_model()
        .with_timestamps()
        .with_soft_delete())
    
    if extra_fields:
        for f in extra_fields:
            builder.add_field(f.name, f.field_type)
    
    return builder.build()


def create_post_model(author_model: str = "User") -> DatabaseModel:
    """Create a standard Post/Article model."""
    return (ModelBuilder("Post", "Blog post or article")
        .add_string("title", max_length=200)
        .add_slug("slug")
        .add_text("content")
        .add_text("excerpt", required=False, nullable=True)
        .add_enum("status", ["draft", "published", "archived"], default="draft")
        .add_integer("view_count", default=0, required=False)
        .add_datetime("published_at", required=False, nullable=True)
        .belongs_to("author", author_model, back_populates="posts")
        .with_timestamps()
        .build())


def create_category_model() -> DatabaseModel:
    """Create a standard Category model."""
    return (ModelBuilder("Category", "Content category")
        .add_string("name", max_length=100, unique=True)
        .add_slug("slug")
        .add_text("description", required=False, nullable=True)
        .add_integer("parent_id", required=False, nullable=True)
        .has_many("posts", "Post", back_populates="category")
        .with_timestamps()
        .build())


def create_product_model() -> DatabaseModel:
    """Create an e-commerce Product model."""
    return (ModelBuilder("Product", "E-commerce product")
        .add_string("name", max_length=200)
        .add_slug("slug")
        .add_text("description")
        .add_float("price")
        .add_float("discount_price", required=False, nullable=True)
        .add_integer("stock", default=0)
        .add_string("sku", max_length=50, unique=True)
        .add_image("image", required=False, nullable=True)
        .add_enum("status", ["active", "inactive", "out_of_stock"], default="active")
        .add_boolean("featured", default=False)
        .with_timestamps()
        .with_soft_delete()
        .build())
