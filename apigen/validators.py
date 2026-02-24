"""
APIGen - Validation System
Comprehensive input validation for models, fields, configs and project structure.
"""

import re
import os
import keyword
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# VALIDATION RESULT
# ============================================================

class Severity(Enum):
    """Validation message severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """Single validation message."""
    severity: Severity
    code: str
    message: str
    field_name: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        prefix = {
            Severity.ERROR: "❌ ERROR",
            Severity.WARNING: "⚠️  WARN",
            Severity.INFO: "ℹ️  INFO",
        }[self.severity]
        
        location = f" [{self.field_name}]" if self.field_name else ""
        hint = f"\n   💡 Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"{prefix}{location}: {self.message}{hint}"


@dataclass
class ValidationResult:
    """Collection of validation messages."""
    messages: List[ValidationMessage] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if no errors exist."""
        return not any(m.severity == Severity.ERROR for m in self.messages)
    
    @property
    def errors(self) -> List[ValidationMessage]:
        """Get only error messages."""
        return [m for m in self.messages if m.severity == Severity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationMessage]:
        """Get only warning messages."""
        return [m for m in self.messages if m.severity == Severity.WARNING]
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def add_error(self, code: str, message: str, 
                  field_name: str = None, suggestion: str = None):
        """Add an error message."""
        self.messages.append(ValidationMessage(
            Severity.ERROR, code, message, field_name, suggestion
        ))
    
    def add_warning(self, code: str, message: str,
                    field_name: str = None, suggestion: str = None):
        """Add a warning message."""
        self.messages.append(ValidationMessage(
            Severity.WARNING, code, message, field_name, suggestion
        ))
    
    def add_info(self, code: str, message: str,
                 field_name: str = None, suggestion: str = None):
        """Add an info message."""
        self.messages.append(ValidationMessage(
            Severity.INFO, code, message, field_name, suggestion
        ))
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.messages.extend(other.messages)
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        lines = []
        lines.append("=" * 50)
        lines.append("  VALIDATION REPORT")
        lines.append("=" * 50)
        
        if self.is_valid:
            lines.append("  ✅ All validations passed!")
        else:
            lines.append(f"  ❌ Found {self.error_count} error(s)")
        
        if self.warning_count > 0:
            lines.append(f"  ⚠️  Found {self.warning_count} warning(s)")
        
        lines.append("-" * 50)
        
        for msg in self.messages:
            lines.append(f"  {msg}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# NAME VALIDATORS
# ============================================================

# Python reserved keywords that can't be used as names
PYTHON_RESERVED: Set[str] = set(keyword.kwlist)

# SQLAlchemy / database reserved words
SQL_RESERVED: Set[str] = {
    'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
    'table', 'column', 'index', 'constraint', 'primary', 'foreign',
    'key', 'references', 'join', 'inner', 'outer', 'left', 'right',
    'where', 'group', 'order', 'having', 'limit', 'offset', 'union',
    'distinct', 'as', 'on', 'and', 'or', 'not', 'null', 'true', 'false',
    'between', 'like', 'in', 'exists', 'case', 'when', 'then', 'else',
    'end', 'count', 'sum', 'avg', 'min', 'max', 'all', 'any', 'some',
    'database', 'schema', 'grant', 'revoke', 'commit', 'rollback',
    'transaction', 'begin', 'declare', 'cursor', 'fetch', 'into',
    'values', 'set', 'from', 'user', 'role', 'view', 'trigger',
    'procedure', 'function', 'exec', 'execute',
}

# FastAPI / Pydantic reserved names
FASTAPI_RESERVED: Set[str] = {
    'request', 'response', 'app', 'router', 'middleware',
    'dependency', 'background', 'websocket', 'config',
    'basemodel', 'field', 'validator', 'root_validator',
    'base', 'metadata', 'query', 'body', 'path', 'header', 'cookie',
}

# Common model names that might cause conflicts
RISKY_MODEL_NAMES: Set[str] = {
    'type', 'model', 'object', 'list', 'dict', 'set', 'str', 'int',
    'float', 'bool', 'bytes', 'tuple', 'base', 'session', 'engine',
    'connection', 'metadata', 'table', 'column', 'query',
}


class NameValidator:
    """Validates names for models, fields, and projects."""
    
    # Valid Python identifier pattern
    IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    # PascalCase pattern for model names
    PASCAL_CASE_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
    
    # snake_case pattern for field names
    SNAKE_CASE_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
    
    # Project name pattern (allows hyphens)
    PROJECT_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_-]*$')
    
    @classmethod
    def validate_model_name(cls, name: str, result: ValidationResult) -> bool:
        """Validate a model/class name."""
        if not name:
            result.add_error("E001", "Model name cannot be empty", "model_name")
            return False
        
        if len(name) < 2:
            result.add_error(
                "E002", f"Model name '{name}' is too short (min 2 chars)",
                "model_name", "Use descriptive names like 'User', 'Product', 'BlogPost'"
            )
            return False
        
        if len(name) > 64:
            result.add_error(
                "E003", f"Model name '{name}' is too long (max 64 chars)",
                "model_name"
            )
            return False
        
        if not cls.IDENTIFIER_PATTERN.match(name):
            result.add_error(
                "E004", f"Model name '{name}' is not a valid Python identifier",
                "model_name", "Use only letters, numbers, and underscores"
            )
            return False
        
        if not cls.PASCAL_CASE_PATTERN.match(name):
            result.add_warning(
                "W001", f"Model name '{name}' should be PascalCase",
                "model_name",
                f"Suggested: '{cls._to_pascal_case(name)}'"
            )
        
        name_lower = name.lower()
        
        if name_lower in PYTHON_RESERVED:
            result.add_error(
                "E005", f"'{name}' is a Python reserved keyword",
                "model_name", f"Try '{name}Item' or '{name}Model' instead"
            )
            return False
        
        if name_lower in RISKY_MODEL_NAMES:
            result.add_warning(
                "W002", f"'{name}' may conflict with built-in types",
                "model_name", f"Consider using a more specific name like 'App{name}'"
            )
        
        if name.endswith('s') and not name.endswith('ss'):
            result.add_warning(
                "W003", f"Model name '{name}' appears to be plural",
                "model_name",
                f"Model names should be singular. Suggested: '{name.rstrip('s')}'"
            )
        
        return True
    
    @classmethod
    def validate_field_name(cls, name: str, model_name: str,
                            result: ValidationResult) -> bool:
        """Validate a field/column name."""
        if not name:
            result.add_error(
                "E010", "Field name cannot be empty",
                f"{model_name}.field_name"
            )
            return False
        
        if len(name) > 64:
            result.add_error(
                "E011", f"Field name '{name}' is too long (max 64 chars)",
                f"{model_name}.{name}"
            )
            return False
        
        if not cls.IDENTIFIER_PATTERN.match(name):
            result.add_error(
                "E012", f"Field name '{name}' is not a valid Python identifier",
                f"{model_name}.{name}"
            )
            return False
        
        if not cls.SNAKE_CASE_PATTERN.match(name):
            result.add_warning(
                "W010", f"Field name '{name}' should be snake_case",
                f"{model_name}.{name}",
                f"Suggested: '{cls._to_snake_case(name)}'"
            )
        
        if name.lower() in PYTHON_RESERVED:
            result.add_error(
                "E013", f"Field '{name}' is a Python reserved keyword",
                f"{model_name}.{name}",
                f"Try '{name}_value' or '{name}_field' instead"
            )
            return False
        
        if name.lower() in SQL_RESERVED:
            result.add_warning(
                "W011", f"Field '{name}' is a SQL reserved word",
                f"{model_name}.{name}",
                "This may cause issues with some databases"
            )
        
        if name.startswith('_'):
            result.add_warning(
                "W012", f"Field '{name}' starts with underscore",
                f"{model_name}.{name}",
                "Leading underscores are typically reserved for internal use"
            )
        
        return True
    
    @classmethod
    def validate_project_name(cls, name: str, result: ValidationResult) -> bool:
        """Validate a project name."""
        if not name:
            result.add_error("E020", "Project name cannot be empty", "project_name")
            return False
        
        if len(name) < 2:
            result.add_error(
                "E021", f"Project name '{name}' is too short (min 2 chars)",
                "project_name"
            )
            return False
        
        if len(name) > 100:
            result.add_error(
                "E022", f"Project name '{name}' is too long (max 100 chars)",
                "project_name"
            )
            return False
        
        if not cls.PROJECT_NAME_PATTERN.match(name):
            result.add_error(
                "E023", f"Project name '{name}' has invalid characters",
                "project_name",
                "Use only lowercase letters, numbers, hyphens, and underscores"
            )
            return False
        
        if name.lower() in PYTHON_RESERVED:
            result.add_error(
                "E024", f"Project name '{name}' is a Python reserved keyword",
                "project_name"
            )
            return False
        
        return True
    
    @classmethod
    def validate_endpoint_path(cls, path: str, result: ValidationResult) -> bool:
        """Validate an API endpoint path."""
        if not path:
            result.add_error("E030", "Endpoint path cannot be empty", "path")
            return False
        
        if not path.startswith('/'):
            result.add_warning(
                "W030", f"Path '{path}' should start with '/'",
                "path", f"Suggested: '/{path}'"
            )
        
        valid_path = re.compile(r'^(/[a-z0-9_-]+|\{[a-z_][a-z0-9_]*\})*/?$', re.IGNORECASE)
        clean_path = path if path.startswith('/') else f'/{path}'
        
        if not valid_path.match(clean_path):
            result.add_warning(
                "W031", f"Path '{path}' may contain invalid characters",
                "path", "Use lowercase letters, numbers, hyphens in paths"
            )
        
        return True
    
    @staticmethod
    def _to_pascal_case(name: str) -> str:
        """Convert string to PascalCase."""
        if '_' in name:
            return ''.join(w.capitalize() for w in name.split('_'))
        if '-' in name:
            return ''.join(w.capitalize() for w in name.split('-'))
        return name[0].upper() + name[1:] if name else name
    
    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert string to snake_case."""
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# ============================================================
# MODEL VALIDATOR
# ============================================================

class ModelValidator:
    """Validates complete model definitions."""
    
    # Sensible limits
    MAX_FIELDS_PER_MODEL = 100
    MAX_RELATIONSHIPS_PER_MODEL = 20
    MAX_INDEXES_PER_MODEL = 30
    
    @classmethod
    def validate(cls, model_dict: Dict[str, Any]) -> ValidationResult:
        """Validate a complete model definition dict."""
        result = ValidationResult()
        
        # Validate model name
        name = model_dict.get('name', '')
        NameValidator.validate_model_name(name, result)
        
        # Validate fields
        fields = model_dict.get('fields', [])
        cls._validate_fields(name, fields, result)
        
        # Validate relationships
        relationships = model_dict.get('relationships', [])
        cls._validate_relationships(name, relationships, result)
        
        # Check model-level constraints
        cls._validate_model_constraints(model_dict, result)
        
        return result
    
    @classmethod
    def validate_model_object(cls, model) -> ValidationResult:
        """Validate a DatabaseModel object."""
        result = ValidationResult()
        
        NameValidator.validate_model_name(model.name, result)
        
        if len(model.fields) == 0:
            result.add_error(
                "E100", f"Model '{model.name}' has no fields",
                model.name, "Add at least one field to the model"
            )
        
        if len(model.fields) > cls.MAX_FIELDS_PER_MODEL:
            result.add_warning(
                "W100", f"Model '{model.name}' has {len(model.fields)} fields (max recommended: {cls.MAX_FIELDS_PER_MODEL})",
                model.name, "Consider splitting into multiple models"
            )
        
        # Check for duplicate field names
        field_names = [f.name for f in model.fields]
        duplicates = [n for n in field_names if field_names.count(n) > 1]
        if duplicates:
            result.add_error(
                "E101", f"Duplicate field names in '{model.name}': {set(duplicates)}",
                model.name
            )
        
        # Validate each field
        for f in model.fields:
            NameValidator.validate_field_name(f.name, model.name, result)
        
        # Validate relationships
        if len(model.relationships) > cls.MAX_RELATIONSHIPS_PER_MODEL:
            result.add_warning(
                "W101", f"Model '{model.name}' has too many relationships ({len(model.relationships)})",
                model.name
            )
        
        # Check auth model has required fields
        if model.is_auth_model:
            auth_field_names = {f.name for f in model.fields}
            required_auth = {'email', 'hashed_password'}
            missing = required_auth - auth_field_names
            if missing:
                result.add_error(
                    "E102", f"Auth model '{model.name}' missing required fields: {missing}",
                    model.name
                )
        
        # Info about auto-generated features
        if model.timestamps:
            result.add_info("I100", f"Timestamps enabled for '{model.name}'", model.name)
        if model.soft_delete:
            result.add_info("I101", f"Soft delete enabled for '{model.name}'", model.name)
        
        return result
    
    @classmethod
    def _validate_fields(cls, model_name: str, fields: List[Dict],
                         result: ValidationResult):
        """Validate field definitions."""
        if not fields:
            result.add_warning(
                "W110", f"Model '{model_name}' has no custom fields defined",
                model_name, "Add fields to make the model useful"
            )
            return
        
        field_names = set()
        has_primary_key = False
        
        for idx, f in enumerate(fields):
            fname = f.get('name', '')
            
            if not fname:
                result.add_error(
                    "E110", f"Field #{idx + 1} in '{model_name}' has no name",
                    model_name
                )
                continue
            
            # Check duplicates
            if fname in field_names:
                result.add_error(
                    "E111", f"Duplicate field '{fname}' in '{model_name}'",
                    f"{model_name}.{fname}"
                )
            field_names.add(fname)
            
            # Validate name
            NameValidator.validate_field_name(fname, model_name, result)
            
            # Validate type
            ftype = f.get('type', f.get('field_type', ''))
            if ftype:
                valid_types = {
                    'string', 'text', 'integer', 'float', 'boolean',
                    'datetime', 'date', 'email', 'url', 'uuid', 'json',
                    'enum', 'file', 'image', 'password', 'phone', 'slug',
                    'ip_address'
                }
                if isinstance(ftype, str) and ftype.lower() not in valid_types:
                    result.add_error(
                        "E112", f"Unknown field type '{ftype}' for '{model_name}.{fname}'",
                        f"{model_name}.{fname}",
                        f"Valid types: {', '.join(sorted(valid_types))}"
                    )
            
            # Check max_length for strings
            max_len = f.get('max_length', 255)
            if isinstance(max_len, int) and max_len > 10000:
                result.add_warning(
                    "W111", f"Very large max_length ({max_len}) for '{model_name}.{fname}'",
                    f"{model_name}.{fname}",
                    "Consider using 'text' type instead"
                )
            
            # Check enum values
            if ftype in ('enum', 'Enum') and not f.get('enum_values'):
                result.add_error(
                    "E113", f"Enum field '{model_name}.{fname}' has no values defined",
                    f"{model_name}.{fname}",
                    "Add enum_values=['value1', 'value2', ...]"
                )
            
            if f.get('primary_key'):
                has_primary_key = True
    
    @classmethod
    def _validate_relationships(cls, model_name: str, relationships: List[Dict],
                                result: ValidationResult):
        """Validate relationship definitions."""
        rel_names = set()
        
        for rel in relationships:
            rname = rel.get('name', '')
            
            if not rname:
                result.add_error(
                    "E120", f"Relationship in '{model_name}' has no name",
                    model_name
                )
                continue
            
            if rname in rel_names:
                result.add_error(
                    "E121", f"Duplicate relationship '{rname}' in '{model_name}'",
                    f"{model_name}.{rname}"
                )
            rel_names.add(rname)
            
            # Check target model
            target = rel.get('target_model', rel.get('target', ''))
            if not target:
                result.add_error(
                    "E122", f"Relationship '{rname}' has no target model",
                    f"{model_name}.{rname}"
                )
            
            # Check relation type
            rel_type = rel.get('relation_type', rel.get('type', ''))
            valid_rel_types = {
                'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
            }
            if isinstance(rel_type, str) and rel_type.lower() not in valid_rel_types:
                result.add_error(
                    "E123", f"Invalid relationship type '{rel_type}'",
                    f"{model_name}.{rname}",
                    f"Valid types: {', '.join(valid_rel_types)}"
                )
    
    @classmethod
    def _validate_model_constraints(cls, model_dict: Dict, result: ValidationResult):
        """Validate model-level constraints."""
        name = model_dict.get('name', 'Unknown')
        
        # Check table name if custom
        table_name = model_dict.get('table_name')
        if table_name:
            if not re.match(r'^[a-z][a-z0-9_]*$', table_name):
                result.add_error(
                    "E130", f"Invalid table name '{table_name}' for model '{name}'",
                    f"{name}.table_name",
                    "Table names should be lowercase with underscores"
                )


# ============================================================
# CONFIG VALIDATOR
# ============================================================

class ConfigValidator:
    """Validates project configuration."""
    
    VALID_DB_TYPES = {'postgresql', 'mysql', 'sqlite', 'mongodb'}
    VALID_AUTH_TYPES = {'jwt', 'session', 'oauth2', 'api_key', 'none'}
    VALID_CACHE_TYPES = {'redis', 'memcached', 'memory', 'none'}
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> ValidationResult:
        """Validate project configuration."""
        result = ValidationResult()
        
        # Project name
        project_name = config.get('project_name', config.get('name', ''))
        if project_name:
            NameValidator.validate_project_name(project_name, result)
        else:
            result.add_error("E200", "Project name is required", "project_name")
        
        # Database type
        db_type = config.get('database', config.get('db_type', 'postgresql'))
        if isinstance(db_type, str) and db_type.lower() not in cls.VALID_DB_TYPES:
            result.add_error(
                "E201", f"Unsupported database type '{db_type}'",
                "database",
                f"Supported: {', '.join(cls.VALID_DB_TYPES)}"
            )
        
        # Auth type
        auth = config.get('auth', config.get('auth_type', 'jwt'))
        if isinstance(auth, str) and auth.lower() not in cls.VALID_AUTH_TYPES:
            result.add_error(
                "E202", f"Unsupported auth type '{auth}'",
                "auth",
                f"Supported: {', '.join(cls.VALID_AUTH_TYPES)}"
            )
        
        # Cache type
        cache = config.get('cache', config.get('cache_type', 'none'))
        if isinstance(cache, str) and cache.lower() not in cls.VALID_CACHE_TYPES:
            result.add_warning(
                "W200", f"Unknown cache type '{cache}'",
                "cache",
                f"Supported: {', '.join(cls.VALID_CACHE_TYPES)}"
            )
        
        # Port
        port = config.get('port', 8000)
        if isinstance(port, int):
            if port < 1 or port > 65535:
                result.add_error(
                    "E203", f"Invalid port number {port}",
                    "port", "Port must be between 1 and 65535"
                )
            elif port < 1024:
                result.add_warning(
                    "W201", f"Port {port} requires root/admin privileges",
                    "port", "Use port >= 1024 for development"
                )
        
        # Models validation
        models = config.get('models', [])
        if not models:
            result.add_warning(
                "W202", "No models defined in configuration",
                "models", "Add at least one model to generate API endpoints"
            )
        
        # Check for duplicate model names
        model_names = []
        for m in models:
            if isinstance(m, dict):
                model_names.append(m.get('name', ''))
            elif hasattr(m, 'name'):
                model_names.append(m.name)
        
        duplicates = [n for n in model_names if model_names.count(n) > 1 and n]
        if duplicates:
            result.add_error(
                "E204", f"Duplicate model names: {set(duplicates)}",
                "models"
            )
        
        # Validate each model
        for m in models:
            if isinstance(m, dict):
                model_result = ModelValidator.validate(m)
                result.merge(model_result)
        
        return result
    
    @classmethod
    def validate_config_object(cls, config) -> ValidationResult:
        """Validate a ProjectConfig object."""
        result = ValidationResult()
        
        NameValidator.validate_project_name(config.project_name, result)
        
        if config.database not in cls.VALID_DB_TYPES:
            result.add_error(
                "E210", f"Unsupported database: {config.database}",
                "database"
            )
        
        if hasattr(config, 'auth') and config.auth not in cls.VALID_AUTH_TYPES:
            result.add_error(
                "E211", f"Unsupported auth type: {config.auth}",
                "auth"
            )
        
        if not config.models:
            result.add_warning(
                "W210", "No models in project config",
                "models"
            )
        
        return result


# ============================================================
# FILE SYSTEM VALIDATOR
# ============================================================

class FileSystemValidator:
    """Validates file system paths and project structure."""
    
    @classmethod
    def validate_output_path(cls, path: str) -> ValidationResult:
        """Validate output directory path."""
        result = ValidationResult()
        
        if not path:
            result.add_error("E300", "Output path cannot be empty", "path")
            return result
        
        # Check if path is absolute or relative
        abs_path = os.path.abspath(path)
        
        # Check parent directory exists
        parent = os.path.dirname(abs_path)
        if parent and not os.path.exists(parent):
            result.add_warning(
                "W300", f"Parent directory does not exist: {parent}",
                "path", "It will be created automatically"
            )
        
        # Check if target already exists
        if os.path.exists(abs_path):
            if os.path.isfile(abs_path):
                result.add_error(
                    "E301", f"Path '{path}' exists but is a file, not a directory",
                    "path"
                )
            elif os.listdir(abs_path):
                result.add_warning(
                    "W301", f"Directory '{path}' already exists and is not empty",
                    "path", "Existing files may be overwritten"
                )
        
        # Check path length
        if len(abs_path) > 255:
            result.add_warning(
                "W302", "Path is very long, may cause issues on some systems",
                "path"
            )
        
        # Check for spaces in path
        if ' ' in abs_path:
            result.add_warning(
                "W303", "Path contains spaces",
                "path", "Paths without spaces are recommended"
            )
        
        return result
    
    @classmethod
    def validate_project_structure(cls, project_dir: str) -> ValidationResult:
        """Validate an existing generated project structure."""
        result = ValidationResult()
        
        if not os.path.exists(project_dir):
            result.add_error("E310", f"Project directory not found: {project_dir}", "path")
            return result
        
        # Expected files/dirs
        expected_files = [
            'main.py',
            'requirements.txt',
            '.env.example',
        ]
        
        expected_dirs = [
            'app',
            'app/models',
            'app/schemas',
            'app/routers',
        ]
        
        for f in expected_files:
            fpath = os.path.join(project_dir, f)
            if not os.path.isfile(fpath):
                result.add_warning(
                    "W310", f"Expected file not found: {f}",
                    "structure"
                )
        
        for d in expected_dirs:
            dpath = os.path.join(project_dir, d)
            if not os.path.isdir(dpath):
                result.add_warning(
                    "W311", f"Expected directory not found: {d}",
                    "structure"
                )
        
        if result.is_valid:
            result.add_info("I310", "Project structure looks valid", "structure")
        
        return result


# ============================================================
# MASTER VALIDATOR (ALL-IN-ONE)
# ============================================================

class ProjectValidator:
    """Master validator - validates entire project configuration."""
    
    @classmethod
    def validate_all(cls, config: Dict[str, Any],
                     output_path: str = None) -> ValidationResult:
        """Run all validations on project configuration."""
        result = ValidationResult()
        
        # 1. Config validation
        config_result = ConfigValidator.validate(config)
        result.merge(config_result)
        
        # 2. Output path validation
        if output_path:
            path_result = FileSystemValidator.validate_output_path(output_path)
            result.merge(path_result)
        
        # 3. Cross-model validation
        cls._validate_cross_model_references(config, result)
        
        return result
    
    @classmethod
    def _validate_cross_model_references(cls, config: Dict, result: ValidationResult):
        """Validate that all model references are valid."""
        models = config.get('models', [])
        
        model_names = set()
        for m in models:
            if isinstance(m, dict):
                model_names.add(m.get('name', ''))
            elif hasattr(m, 'name'):
                model_names.add(m.name)
        
        # Check relationship targets
        for m in models:
            if isinstance(m, dict):
                mname = m.get('name', '')
                for rel in m.get('relationships', []):
                    target = rel.get('target_model', rel.get('target', ''))
                    if target and target not in model_names:
                        result.add_warning(
                            "W400",
                            f"Model '{mname}' references '{target}' which is not defined",
                            f"{mname}.relationships",
                            f"Make sure '{target}' model exists or will be created"
                        )


# ============================================================
# QUICK VALIDATION FUNCTIONS
# ============================================================

def validate_model_name(name: str) -> Tuple[bool, str]:
    """Quick validation for model name. Returns (is_valid, message)."""
    result = ValidationResult()
    NameValidator.validate_model_name(name, result)
    
    if result.is_valid:
        return True, f"✅ '{name}' is a valid model name"
    else:
        errors = "; ".join(e.message for e in result.errors)
        return False, f"❌ {errors}"


def validate_field_name(name: str, model: str = "Model") -> Tuple[bool, str]:
    """Quick validation for field name."""
    result = ValidationResult()
    NameValidator.validate_field_name(name, model, result)
    
    if result.is_valid:
        return True, f"✅ '{name}' is a valid field name"
    else:
        errors = "; ".join(e.message for e in result.errors)
        return False, f"❌ {errors}"


def validate_project_config(config: Dict) -> ValidationResult:
    """Validate project configuration dict."""
    return ConfigValidator.validate(config)


def validate_quick(name: str, name_type: str = "model") -> bool:
    """Quick boolean check for name validity."""
    result = ValidationResult()
    
    if name_type == "model":
        NameValidator.validate_model_name(name, result)
    elif name_type == "field":
        NameValidator.validate_field_name(name, "Model", result)
    elif name_type == "project":
        NameValidator.validate_project_name(name, result)
    
    return result.is_valid
