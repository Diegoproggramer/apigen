# File: apigen/utils.py
"""
NexaFlow APIGen - Utility Functions & Helpers
===============================================
High-performance string transformation, file I/O, and code-formatting
utilities used throughout the generation pipeline.

Performance strategy:
- ALL string-conversion functions are decorated with ``@lru_cache(maxsize=None)``
  so repeated calls (which happen thousands of times during 50K-line generation)
  are amortised to O(1) after first invocation.
- File I/O helpers use buffered writes and atomic rename for safety.
- No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("apigen.utils")

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (compiled once at module load)
# ---------------------------------------------------------------------------

_CAMEL_TO_SNAKE_RE1: re.Pattern[str] = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_TO_SNAKE_RE2: re.Pattern[str] = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALPHANUM_RE: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9]")
_MULTI_UNDERSCORE_RE: re.Pattern[str] = re.compile(r"_{2,}")
_LEADING_TRAILING_UNDERSCORE_RE: re.Pattern[str] = re.compile(r"^_+|_+$")
_SPLIT_WORDS_RE: re.Pattern[str] = re.compile(
    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]|\d+"
)

# Python keywords that cannot be used as identifiers
_PYTHON_KEYWORDS: FrozenSet[str] = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else",
    "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield",
})

# Common Python builtins that may clash with column/table names
_PYTHON_BUILTINS: FrozenSet[str] = frozenset({
    "id", "type", "list", "dict", "set", "str", "int", "float",
    "bool", "bytes", "object", "hash", "input", "print", "range",
    "len", "map", "filter", "zip", "enumerate", "super", "property",
    "format", "iter", "next", "open", "exec", "eval", "compile",
    "vars", "dir", "help", "repr", "staticmethod", "classmethod",
    "isinstance", "issubclass", "callable", "getattr", "setattr",
    "delattr", "hasattr", "abs", "all", "any", "bin", "chr", "hex",
    "max", "min", "oct", "ord", "pow", "round", "sorted", "sum",
})


# ---------------------------------------------------------------------------
# Cached string transformation functions
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def to_snake_case(name: str) -> str:
    """
    Convert any string to snake_case.

    Examples:
        >>> to_snake_case("UserProfile")
        'user_profile'
        >>> to_snake_case("getHTTPResponse")
        'get_http_response'
        >>> to_snake_case("already_snake")
        'already_snake'
        >>> to_snake_case("XML2JSON")
        'xml2_json'

    First call: O(n) where n = len(name).
    Subsequent calls with same input: O(1) via LRU cache.
    """
    if not name:
        return ""
    s: str = _CAMEL_TO_SNAKE_RE1.sub(r"\1_\2", name)
    s = _CAMEL_TO_SNAKE_RE2.sub(r"\1_\2", s)
    s = _NON_ALPHANUM_RE.sub("_", s)
    s = _MULTI_UNDERSCORE_RE.sub("_", s)
    s = _LEADING_TRAILING_UNDERSCORE_RE.sub("", s)
    return s.lower()


@functools.lru_cache(maxsize=None)
def to_pascal_case(name: str) -> str:
    """
    Convert any string to PascalCase.

    Examples:
        >>> to_pascal_case("user_profile")
        'UserProfile'
        >>> to_pascal_case("http_response")
        'HttpResponse'
        >>> to_pascal_case("AlreadyPascal")
        'Alreadypascal'  # re-normalises

    First call: O(n).  Subsequent: O(1).
    """
    if not name:
        return ""
    words: List[str] = _extract_words(name)
    return "".join(word.capitalize() for word in words)


@functools.lru_cache(maxsize=None)
def to_camel_case(name: str) -> str:
    """
    Convert any string to camelCase.

    Examples:
        >>> to_camel_case("user_profile")
        'userProfile'
        >>> to_camel_case("HTTPResponse")
        'httpResponse'

    First call: O(n).  Subsequent: O(1).
    """
    if not name:
        return ""
    words: List[str] = _extract_words(name)
    if not words:
        return ""
    first: str = words[0].lower()
    rest: str = "".join(w.capitalize() for w in words[1:])
    return first + rest


@functools.lru_cache(maxsize=None)
def to_kebab_case(name: str) -> str:
    """
    Convert any string to kebab-case (used in URL paths).

    First call: O(n).  Subsequent: O(1).
    """
    if not name:
        return ""
    words: List[str] = _extract_words(name)
    return "-".join(w.lower() for w in words)


@functools.lru_cache(maxsize=None)
def to_title_human(name: str) -> str:
    """
    Convert identifier to human-readable title.

    Examples:
        >>> to_title_human("user_profile")
        'User Profile'
        >>> to_title_human("orderItem")
        'Order Item'
    """
    if not name:
        return ""
    words: List[str] = _extract_words(name)
    return " ".join(w.capitalize() for w in words)


@functools.lru_cache(maxsize=None)
def to_plural(name: str) -> str:
    """
    Naive English pluralisation sufficient for code generation.

    Handles common suffixes. For production i18n, use a dedicated library.

    First call: O(n).  Subsequent: O(1).
    """
    if not name:
        return ""

    lower: str = name.lower()

    # Irregular common words in DB schemas
    irregulars: Dict[str, str] = {
        "person": "people",
        "child": "children",
        "man": "men",
        "woman": "women",
        "mouse": "mice",
        "goose": "geese",
        "tooth": "teeth",
        "foot": "feet",
        "datum": "data",
        "index": "indices",
        "matrix": "matrices",
        "vertex": "vertices",
        "axis": "axes",
        "crisis": "crises",
        "analysis": "analyses",
        "status": "statuses",
        "address": "addresses",
    }

    if lower in irregulars:
        plural: str = irregulars[lower]
        # Preserve original casing of first char
        if name[0].isupper():
            return plural[0].upper() + plural[1:]
        return plural

    # Already plural-looking (very naive)
    if lower.endswith("s") and not lower.endswith("ss"):
        return name

    # Rules ordered by specificity
    if lower.endswith(("sh", "ch", "x", "z", "ss")):
        return name + "es"
    if lower.endswith("y") and len(name) > 1 and lower[-2] not in "aeiou":
        return name[:-1] + "ies"
    if lower.endswith("f"):
        return name[:-1] + "ves"
    if lower.endswith("fe"):
        return name[:-2] + "ves"
    if lower.endswith("o") and len(name) > 1 and lower[-2] not in "aeiou":
        return name + "es"

    return name + "s"


@functools.lru_cache(maxsize=None)
def to_singular(name: str) -> str:
    """
    Naive English singularisation (reverse of to_plural).

    First call: O(n).  Subsequent: O(1).
    """
    if not name:
        return ""

    lower: str = name.lower()

    # Reverse irregulars
    reverse_irregulars: Dict[str, str] = {
        "people": "person",
        "children": "child",
        "men": "man",
        "women": "woman",
        "mice": "mouse",
        "geese": "goose",
        "teeth": "tooth",
        "feet": "foot",
        "data": "datum",
        "indices": "index",
        "matrices": "matrix",
        "vertices": "vertex",
        "axes": "axis",
        "crises": "crisis",
        "analyses": "analysis",
        "statuses": "status",
        "addresses": "address",
    }

    if lower in reverse_irregulars:
        singular: str = reverse_irregulars[lower]
        if name[0].isupper():
            return singular[0].upper() + singular[1:]
        return singular

    # Rules in reverse order of pluralisation
    if lower.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if lower.endswith("ves"):
        return name[:-3] + "f"
    if lower.endswith("oes") and len(name) > 3:
        return name[:-2]
    if lower.endswith("ses") or lower.endswith("xes") or lower.endswith("zes"):
        return name[:-2]
    if lower.endswith("ches") or lower.endswith("shes"):
        return name[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return name[:-1]

    return name


@functools.lru_cache(maxsize=None)
def _extract_words(name: str) -> Tuple[str, ...]:
    """
    Extract individual words from any casing style.

    Returns a tuple (hashable for LRU cache) of lowercase word strings.

    Complexity: O(n) first call, O(1) subsequent.
    """
    # First replace non-alphanumeric with spaces
    cleaned: str = _NON_ALPHANUM_RE.sub(" ", name)
    # Find word boundaries via regex
    words: List[str] = _SPLIT_WORDS_RE.findall(cleaned)
    return tuple(w.lower() for w in words if w)


@functools.lru_cache(maxsize=None)
def safe_identifier(name: str) -> str:
    """
    Ensure a string is a safe Python identifier.

    - Converts to snake_case
    - Prefixes with underscore if starts with digit
    - Appends underscore if it's a Python keyword or dangerous builtin

    First call: O(n).  Subsequent: O(1).
    """
    result: str = to_snake_case(name)
    if not result:
        return "_unnamed"

    # Cannot start with digit
    if result[0].isdigit():
        result = f"_{result}"

    # Python keyword clash
    if result in _PYTHON_KEYWORDS or result in _PYTHON_BUILTINS:
        result = f"{result}_"

    return result


@functools.lru_cache(maxsize=None)
def table_to_class_name(table_name: str) -> str:
    """Convert a table name to a PascalCase ORM class name."""
    return to_pascal_case(table_name)


@functools.lru_cache(maxsize=None)
def table_to_router_prefix(table_name: str) -> str:
    """Convert a table name to a URL router prefix (kebab, plural)."""
    plural: str = to_plural(table_name)
    return f"/{to_kebab_case(plural)}"


@functools.lru_cache(maxsize=None)
def table_to_router_tag(table_name: str) -> str:
    """Convert a table name to a human-readable router tag."""
    return to_title_human(to_plural(table_name))


@functools.lru_cache(maxsize=None)
def column_to_field_name(column_name: str) -> str:
    """Convert a column name to a safe Python field name."""
    return safe_identifier(column_name)


# ---------------------------------------------------------------------------
# Indentation & code formatting helpers
# ---------------------------------------------------------------------------


def indent(text: str, level: int = 1, size: int = 4) -> str:
    """
    Indent every line of *text* by *level* × *size* spaces.

    Uses str.join for O(n) performance (no repeated concatenation).
    """
    prefix: str = " " * (level * size)
    lines: List[str] = text.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def indent_lines(lines: Sequence[str], level: int = 1, size: int = 4) -> List[str]:
    """Indent a list of lines, returning a new list. O(n)."""
    prefix: str = " " * (level * size)
    return [prefix + line if line.strip() else line for line in lines]


def make_docstring(text: str, indent_level: int = 1, size: int = 4) -> str:
    """
    Create a properly formatted Python docstring.

    Single-line docstrings stay on one line; multi-line use triple-quote blocks.
    """
    prefix: str = " " * (indent_level * size)
    stripped: str = text.strip()

    if "\n" not in stripped and len(stripped) + len(prefix) + 6 <= 99:
        return f'{prefix}"""{stripped}"""'

    doc_lines: List[str] = stripped.split("\n")
    parts: List[str] = [f'{prefix}"""']
    parts.extend(f"{prefix}{line}" for line in doc_lines)
    parts.append(f'{prefix}"""')
    return "\n".join(parts)


def wrap_in_quotes(value: str) -> str:
    """Wrap a string value in double quotes, escaping internals."""
    escaped: str = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_list_literal(items: Sequence[str], quote: bool = True) -> str:
    """
    Format a Python list literal from a sequence of strings.

    If *quote* is True, each item is wrapped in quotes.
    """
    if quote:
        inner: str = ", ".join(f'"{item}"' for item in items)
    else:
        inner = ", ".join(items)
    return f"[{inner}]"


def format_dict_literal(
    mapping: Dict[str, str],
    quote_keys: bool = True,
    quote_values: bool = True,
) -> str:
    """Format a Python dict literal from a string mapping."""
    parts: List[str] = []
    for k, v in mapping.items():
        key_str: str = f'"{k}"' if quote_keys else k
        val_str: str = f'"{v}"' if quote_values else v
        parts.append(f"{key_str}: {val_str}")
    inner: str = ", ".join(parts)
    return "{" + inner + "}"


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def ensure_directory(path: Path) -> None:
    """
    Create directory (and parents) if it doesn't exist.

    Uses os.makedirs with exist_ok for atomic safety.
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)


def write_file(path: Path, content: str, atomic: bool = True) -> int:
    """
    Write *content* to *path*.

    When *atomic* is True, writes to a temporary file first then renames —
    this prevents partial writes on crash.

    Returns the number of bytes written.
    """
    ensure_directory(path.parent)

    encoded: bytes = content.encode("utf-8")
    byte_count: int = len(encoded)

    if atomic:
        fd: int
        tmp_path: str
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            os.write(fd, encoded)
            os.close(fd)
            shutil.move(tmp_path, str(path))
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    else:
        path.write_bytes(encoded)

    logger.debug("Wrote %d bytes to %s", byte_count, path)
    return byte_count


def write_files_batch(
    files: Dict[str, str],
    base_dir: Path,
    atomic: bool = True,
) -> Tuple[int, int]:
    """
    Write multiple files at once.

    Args:
        files: Mapping of relative path → content.
        base_dir: Root output directory.
        atomic: Use atomic writes.

    Returns:
        Tuple of (total_files_written, total_bytes_written).
    """
    total_files: int = 0
    total_bytes: int = 0

    for rel_path, content in files.items():
        full_path: Path = base_dir / rel_path
        byte_count: int = write_file(full_path, content, atomic=atomic)
        total_files += 1
        total_bytes += byte_count

    logger.info(
        "Batch write complete: %d files, %d bytes to %s",
        total_files,
        total_bytes,
        base_dir,
    )
    return total_files, total_bytes


def read_file(path: Path) -> str:
    """Read a file and return its content as a string."""
    return path.read_text(encoding="utf-8")


def clean_directory(path: Path, keep_git: bool = True) -> None:
    """
    Remove all contents of a directory without removing the directory itself.

    If *keep_git* is True, ``.git`` and ``.gitignore`` are preserved.
    """
    if not path.exists():
        return

    for item in path.iterdir():
        if keep_git and item.name in {".git", ".gitignore"}:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    logger.debug("Cleaned directory: %s (keep_git=%s)", path, keep_git)


# ---------------------------------------------------------------------------
# Checksum & metrics
# ---------------------------------------------------------------------------


def sha256_hex(content: str) -> str:
    """Return SHA-256 hex digest of a string. O(n)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def count_lines(content: str) -> int:
    """Count the number of lines in a string. O(n)."""
    if not content:
        return 0
    return content.count("\n") + (1 if not content.endswith("\n") else 0)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """
    Simple context-manager timer for profiling generation steps.

    Usage:
        with Timer("generate models") as t:
            ...
        print(t.elapsed)
    """

    __slots__ = ("label", "start_time", "end_time", "elapsed")

    def __init__(self, label: str = "operation") -> None:
        self.label: str = label
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        logger.info(
            "Timer [%s]: %.4f seconds",
            self.label,
            self.elapsed,
        )

    def __repr__(self) -> str:
        return f"<Timer {self.label}: {self.elapsed:.4f}s>"


# ---------------------------------------------------------------------------
# Import statement builder
# ---------------------------------------------------------------------------


def build_import_block(imports: Dict[str, Set[str]]) -> str:
    """
    Build a sorted, de-duplicated import block from a mapping of
    module → set of names.

    Example:
        >>> build_import_block({"typing": {"List", "Optional"}, "datetime": {"datetime"}})
        'from datetime import datetime\\nfrom typing import List, Optional'

    Complexity: O(M log M + N log N) where M = modules, N = total names.
    """
    lines: List[str] = []
    for module in sorted(imports.keys()):
        names: List[str] = sorted(imports[module])
        if names:
            names_str: str = ", ".join(names)
            lines.append(f"from {module} import {names_str}")
        else:
            lines.append(f"import {module}")
    return "\n".join(lines)


def merge_import_dicts(
    *dicts: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Merge multiple import dictionaries into one, unifying sets.

    Complexity: O(sum of all entries).
    """
    result: Dict[str, Set[str]] = {}
    for d in dicts:
        for module, names in d.items():
            if module in result:
                result[module] |= names
            else:
                result[module] = set(names)
    return result


# ---------------------------------------------------------------------------
# SQLAlchemy type-to-import mapping
# ---------------------------------------------------------------------------

# Maps SQLAlchemy type constructor names to their import modules
SQLALCHEMY_TYPE_IMPORTS: Dict[str, Tuple[str, str]] = {
    "Integer": ("sqlalchemy", "Integer"),
    "BigInteger": ("sqlalchemy", "BigInteger"),
    "SmallInteger": ("sqlalchemy", "SmallInteger"),
    "Float": ("sqlalchemy", "Float"),
    "Numeric": ("sqlalchemy", "Numeric"),
    "Double": ("sqlalchemy", "Double"),
    "Boolean": ("sqlalchemy", "Boolean"),
    "String": ("sqlalchemy", "String"),
    "Text": ("sqlalchemy", "Text"),
    "CHAR": ("sqlalchemy", "CHAR"),
    "LargeBinary": ("sqlalchemy", "LargeBinary"),
    "Date": ("sqlalchemy", "Date"),
    "DateTime": ("sqlalchemy", "DateTime"),
    "Time": ("sqlalchemy", "Time"),
    "Interval": ("sqlalchemy", "Interval"),
    "UUID": ("sqlalchemy.dialects.postgresql", "UUID"),
    "JSON": ("sqlalchemy", "JSON"),
    "JSONB": ("sqlalchemy.dialects.postgresql", "JSONB"),
    "ARRAY": ("sqlalchemy.dialects.postgresql", "ARRAY"),
    "HSTORE": ("sqlalchemy.dialects.postgresql", "HSTORE"),
    "INET": ("sqlalchemy.dialects.postgresql", "INET"),
    "CIDR": ("sqlalchemy.dialects.postgresql", "CIDR"),
    "MACADDR": ("sqlalchemy.dialects.postgresql", "MACADDR"),
    "Enum": ("sqlalchemy", "Enum"),
}

# Maps Python type hint strings to their import modules
PYTHON_TYPE_IMPORTS: Dict[str, Tuple[str, str]] = {
    "date": ("datetime", "date"),
    "datetime": ("datetime", "datetime"),
    "time": ("datetime", "time"),
    "timedelta": ("datetime", "timedelta"),
    "Decimal": ("decimal", "Decimal"),
    "UUID": ("uuid", "UUID"),
}


def collect_column_imports(
    column_type_str: str,
    python_type_hint: str,
) -> Dict[str, Set[str]]:
    """
    Given a SQLAlchemy type string and a Python type hint, return the
    necessary imports as a dict.

    Complexity: O(1) — dictionary lookups only.
    """
    result: Dict[str, Set[str]] = {}

    # SQLAlchemy type import
    # Extract the base type name (e.g. "String(255)" → "String")
    base_sa_type: str = column_type_str.split("(")[0].strip()
    if base_sa_type in SQLALCHEMY_TYPE_IMPORTS:
        module, name = SQLALCHEMY_TYPE_IMPORTS[base_sa_type]
        result.setdefault(module, set()).add(name)

    # Handle SQLEnum special case
    if "SQLEnum" in column_type_str or "Enum" in base_sa_type:
        result.setdefault("sqlalchemy", set()).add("Enum")

    # Python type imports
    # Strip Optional[] wrapper
    clean_hint: str = python_type_hint
    if clean_hint.startswith("Optional["):
        clean_hint = clean_hint[9:-1]
    if clean_hint.startswith("List["):
        clean_hint = clean_hint[5:-1]
    if clean_hint.startswith("Dict["):
        clean_hint = clean_hint[5:-1]

    for type_name, (mod, imp) in PYTHON_TYPE_IMPORTS.items():
        if type_name in python_type_hint:
            result.setdefault(mod, set()).add(imp)

    return result


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "to_snake_case",
    "to_pascal_case",
    "to_camel_case",
    "to_kebab_case",
    "to_title_human",
    "to_plural",
    "to_singular",
    "safe_identifier",
    "table_to_class_name",
    "table_to_router_prefix",
    "table_to_router_tag",
    "column_to_field_name",
    "indent",
    "indent_lines",
    "make_docstring",
    "wrap_in_quotes",
    "format_list_literal",
    "format_dict_literal",
    "ensure_directory",
    "write_file",
    "write_files_batch",
    "read_file",
    "clean_directory",
    "sha256_hex",
    "count_lines",
    "Timer",
    "build_import_block",
    "merge_import_dicts",
    "collect_column_imports",
    "SQLALCHEMY_TYPE_IMPORTS",
    "PYTHON_TYPE_IMPORTS",
]

logger.debug("apigen.utils loaded — %d public symbols.", len(__all__))
