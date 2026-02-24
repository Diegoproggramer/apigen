"""
APIGen - Utility Functions
Helper functions for code generation, file management, and validation.
"""

import os
import re
import json
import keyword
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


# ============================================================
# NAMING CONVENTIONS
# ============================================================

def to_snake_case(name: str) -> str:
    """Convert string to snake_case.
    
    Examples:
        UserProfile -> user_profile
        HTTPResponse -> http_response
        myVariable -> my_variable
    """
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_pascal_case(name: str) -> str:
    """Convert string to PascalCase.
    
    Examples:
        user_profile -> UserProfile
        my-variable -> MyVariable
        hello world -> HelloWorld
    """
    words = re.split(r'[_\-\s]+', name)
    return ''.join(word.capitalize() for word in words)


def to_camel_case(name: str) -> str:
    """Convert string to camelCase.
    
    Examples:
        user_profile -> userProfile
        my-variable -> myVariable
    """
    pascal = to_pascal_case(name)
    return pascal[0].lower() + pascal[1:] if pascal else ''


def to_kebab_case(name: str) -> str:
    """Convert string to kebab-case.
    
    Examples:
        UserProfile -> user-profile
        my_variable -> my-variable
    """
    return to_snake_case(name).replace('_', '-')


def pluralize(word: str) -> str:
    """Simple English pluralization.
    
    Examples:
        user -> users
        category -> categories
        post -> posts
        status -> statuses
    """
    if word.endswith('y') and word[-2:] not in ('ay', 'ey', 'oy', 'uy'):
        return word[:-1] + 'ies'
    elif word.endswith(('s', 'sh', 'ch', 'x', 'z')):
        return word + 'es'
    elif word.endswith('f'):
        return word[:-1] + 'ves'
    elif word.endswith('fe'):
        return word[:-2] + 'ves'
    else:
        return word + 's'


def singularize(word: str) -> str:
    """Simple English singularization."""
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    elif word.endswith('ves'):
        return word[:-3] + 'f'
    elif word.endswith('ses') or word.endswith('xes') or word.endswith('zes'):
        return word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


# ============================================================
# VALIDATION
# ============================================================

def is_valid_python_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return name.isidentifier() and not keyword.iskeyword(name)


def is_valid_project_name(name: str) -> bool:
    """Check if a string is a valid project name."""
    pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
    return bool(re.match(pattern, name)) and len(name) <= 50


def validate_model_name(name: str) -> Tuple[bool, str]:
    """Validate a model name and return (is_valid, error_message)."""
    if not name:
        return False, "Model name cannot be empty"
    if not name[0].isupper():
        return False, "Model name must start with uppercase letter"
    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
        return False, "Model name must be alphanumeric (PascalCase)"
    if name in ('Base', 'Model', 'Session', 'Query', 'Column'):
        return False, f"'{name}' is a reserved SQLAlchemy name"
    return True, ""


def validate_field_type(field_type: str) -> bool:
    """Check if a field type is supported."""
    valid_types = {
        'str', 'string', 'text',
        'int', 'integer',
        'float', 'decimal',
        'bool', 'boolean',
        'date', 'datetime',
        'email', 'url', 'uuid',
        'json', 'list', 'dict',
    }
    return field_type.lower() in valid_types


# ============================================================
# TYPE MAPPING
# ============================================================

SQLALCHEMY_TYPE_MAP = {
    'str': 'String(255)',
    'string': 'String(255)',
    'text': 'Text',
    'int': 'Integer',
    'integer': 'Integer',
    'float': 'Float',
    'decimal': 'Float',
    'bool': 'Boolean',
    'boolean': 'Boolean',
    'date': 'DateTime',
    'datetime': 'DateTime(timezone=True)',
    'email': 'String(320)',
    'url': 'String(2048)',
    'uuid': 'String(36)',
    'json': 'Text',
}

PYDANTIC_TYPE_MAP = {
    'str': 'str',
    'string': 'str',
    'text': 'str',
    'int': 'int',
    'integer': 'int',
    'float': 'float',
    'decimal': 'float',
    'bool': 'bool',
    'boolean': 'bool',
    'date': 'datetime',
    'datetime': 'datetime',
    'email': 'str',
    'url': 'str',
    'uuid': 'str',
    'json': 'dict',
}


def get_sqlalchemy_type(field_type: str) -> str:
    """Convert a simple type to SQLAlchemy column type."""
    return SQLALCHEMY_TYPE_MAP.get(field_type.lower(), 'String(255)')


def get_pydantic_type(field_type: str) -> str:
    """Convert a simple type to Pydantic field type."""
    return PYDANTIC_TYPE_MAP.get(field_type.lower(), 'str')


# ============================================================
# FILE OPERATIONS
# ============================================================

def ensure_directory(path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def write_file(filepath: str, content: str, overwrite: bool = False) -> bool:
    """Write content to a file.
    
    Returns:
        True if file was written, False if skipped.
    """
    path = Path(filepath)
    
    if path.exists() and not overwrite:
        return False
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return True


def read_file(filepath: str) -> Optional[str]:
    """Read file content. Returns None if file doesn't exist."""
    path = Path(filepath)
    if path.exists():
        return path.read_text(encoding='utf-8')
    return None


def load_json_config(filepath: str) -> Optional[Dict]:
    """Load a JSON configuration file."""
    content = read_file(filepath)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


def save_json_config(filepath: str, data: Dict) -> bool:
    """Save data to a JSON configuration file."""
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return write_file(filepath, content, overwrite=True)
    except (TypeError, ValueError):
        return False


# ============================================================
# PROJECT STRUCTURE
# ============================================================

def get_project_structure(project_name: str) -> Dict[str, List[str]]:
    """Get the standard project directory structure."""
    return {
        'directories': [
            project_name,
            f'{project_name}/models',
            f'{project_name}/schemas',
            f'{project_name}/crud',
            f'{project_name}/routes',
            f'{project_name}/core',
            f'{project_name}/tests',
            f'{project_name}/migrations',
        ],
        'init_files': [
            f'{project_name}/models/__init__.py',
            f'{project_name}/schemas/__init__.py',
            f'{project_name}/crud/__init__.py',
            f'{project_name}/routes/__init__.py',
            f'{project_name}/core/__init__.py',
            f'{project_name}/tests/__init__.py',
        ]
    }


def create_project_skeleton(project_name: str) -> List[str]:
    """Create the full project directory skeleton.
    
    Returns:
        List of created paths.
    """
    structure = get_project_structure(project_name)
    created = []
    
    for directory in structure['directories']:
        path = ensure_directory(directory)
        created.append(str(path))
    
    for init_file in structure['init_files']:
        if write_file(init_file, '"""Auto-generated by APIGen."""\n'):
            created.append(init_file)
    
    return created


# ============================================================
# CODE FORMATTING
# ============================================================

def indent(text: str, spaces: int = 4) -> str:
    """Indent each line of text by specified spaces."""
    prefix = ' ' * spaces
    lines = text.split('\n')
    return '\n'.join(prefix + line if line.strip() else line for line in lines)


def generate_imports(modules: List[str]) -> str:
    """Generate sorted import statements."""
    stdlib = []
    third_party = []
    local = []
    
    for module in sorted(set(modules)):
        if module.startswith('.') or module.startswith('from .'):
            local.append(module)
        elif any(module.startswith(f'from {pkg}') or module.startswith(f'import {pkg}')
                 for pkg in ('fastapi', 'sqlalchemy', 'pydantic', 'jose', 'passlib', 'uvicorn')):
            third_party.append(module)
        else:
            stdlib.append(module)
    
    sections = []
    if stdlib:
        sections.append('\n'.join(stdlib))
    if third_party:
        sections.append('\n'.join(third_party))
    if local:
        sections.append('\n'.join(local))
    
    return '\n\n'.join(sections)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_tree(directory: str, prefix: str = "", max_depth: int = 3, _depth: int = 0) -> str:
    """Generate a tree view string of a directory."""
    if _depth >= max_depth:
        return ""
    
    path = Path(directory)
    if not path.exists():
        return f"{prefix}{path.name}/ (not found)\n"
    
    lines = []
    if _depth == 0:
        lines.append(f"{path.name}/\n")
    
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        if item.is_dir() and not item.name.startswith('.'):
            lines.append(f"{prefix}{connector}{item.name}/\n")
            extension = "    " if is_last else "│   "
            lines.append(print_tree(
                str(item), prefix + extension, max_depth, _depth + 1
            ))
        elif item.is_file():
            size = format_file_size(item.stat().st_size)
            lines.append(f"{prefix}{connector}{item.name} ({size})\n")
    
    return ''.join(lines)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
