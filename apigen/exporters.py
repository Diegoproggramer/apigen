"""
APIGen - Export System
Multi-format project exporters: ZIP, Docker, OpenAPI spec, and more.
"""

import os
import json
import shutil
import zipfile
import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod


# ============================================================
# BASE EXPORTER
# ============================================================

class BaseExporter(ABC):
    """Abstract base class for all exporters."""
    
    def __init__(self, project_dir: str, config: Dict[str, Any] = None):
        self.project_dir = os.path.abspath(project_dir)
        self.config = config or {}
        self.project_name = config.get('project_name', 
                                        os.path.basename(project_dir))
        self._exported_files: List[str] = []
    
    @abstractmethod
    def export(self, output_path: str = None) -> str:
        """Export the project. Returns output file/dir path."""
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Return the export format name."""
        pass
    
    def _collect_files(self, exclude_patterns: List[str] = None) -> List[str]:
        """Collect all files in project directory."""
        exclude = exclude_patterns or [
            '__pycache__', '.pyc', '.git', '.DS_Store',
            'node_modules', '.env', '*.egg-info', '.venv', 'venv'
        ]
        
        collected = []
        for root, dirs, files in os.walk(self.project_dir):
            # Filter directories
            dirs[:] = [d for d in dirs 
                      if not any(p.strip('*') in d for p in exclude)]
            
            for f in files:
                if not any(p.strip('*') in f for p in exclude):
                    full_path = os.path.join(root, f)
                    collected.append(full_path)
        
        return collected
    
    def _get_relative_path(self, full_path: str) -> str:
        """Get path relative to project directory."""
        return os.path.relpath(full_path, self.project_dir)
    
    def _ensure_output_dir(self, path: str):
        """Ensure output directory exists."""
        output_dir = os.path.dirname(path) if '.' in os.path.basename(path) else path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


# ============================================================
# ZIP EXPORTER
# ============================================================

class ZipExporter(BaseExporter):
    """Export project as a ZIP archive."""
    
    def get_format_name(self) -> str:
        return "ZIP Archive"
    
    def export(self, output_path: str = None) -> str:
        """Export project as ZIP file."""
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.project_name}_{timestamp}.zip"
        
        if not output_path.endswith('.zip'):
            output_path += '.zip'
        
        self._ensure_output_dir(output_path)
        files = self._collect_files()
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                rel_path = self._get_relative_path(file_path)
                arcname = os.path.join(self.project_name, rel_path)
                zf.write(file_path, arcname)
                self._exported_files.append(rel_path)
        
        return os.path.abspath(output_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ZIP export statistics."""
        return {
            'format': self.get_format_name(),
            'files_count': len(self._exported_files),
            'files': self._exported_files,
        }


# ============================================================
# DOCKER EXPORTER
# ============================================================

class DockerExporter(BaseExporter):
    """Export project with complete Docker setup."""
    
    def get_format_name(self) -> str:
        return "Docker Deployment"
    
    def export(self, output_path: str = None) -> str:
        """Export with Docker configuration."""
        output_dir = output_path or f"{self.project_name}_docker"
        self._ensure_output_dir(output_dir)
        
        # Copy project files
        if os.path.exists(output_dir) and output_dir != self.project_dir:
            shutil.copytree(self.project_dir, output_dir, dirs_exist_ok=True)
        
        # Generate Dockerfile if not exists
        dockerfile_path = os.path.join(output_dir, 'Dockerfile')
        if not os.path.exists(dockerfile_path):
            self._write_dockerfile(dockerfile_path)
        
        # Generate docker-compose.yml if not exists
        compose_path = os.path.join(output_dir, 'docker-compose.yml')
        if not os.path.exists(compose_path):
            self._write_docker_compose(compose_path)
        
        # Generate .dockerignore
        dockerignore_path = os.path.join(output_dir, '.dockerignore')
        if not os.path.exists(dockerignore_path):
            self._write_dockerignore(dockerignore_path)
        
        # Generate deployment scripts
        self._write_deploy_scripts(output_dir)
        
        return os.path.abspath(output_dir)
    
    def _write_dockerfile(self, path: str):
        """Generate production Dockerfile."""
        db_type = self.config.get('database', 'postgresql')
        
        db_packages = {
            'postgresql': 'libpq-dev gcc',
            'mysql': 'default-libmysqlclient-dev gcc',
            'sqlite': '',
        }
        
        sys_deps = db_packages.get(db_type, '')
        
        content = f'''# === APIGen Generated Dockerfile ===
# Project: {self.project_name}
# Generated: {datetime.datetime.now().isoformat()}

# ---------- Build Stage ----------
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies
{f"RUN apt-get update && apt-get install -y {sys_deps} && rm -rf /var/lib/apt/lists/*" if sys_deps else "# No extra system deps needed"}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Production Stage ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

{f"RUN apt-get update && apt-get install -y libpq5 && rm -rf /var/lib/apt/lists/*" if db_type == 'postgresql' else "# No extra runtime deps"}

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
'''
        with open(path, 'w') as f:
            f.write(content)
    
    def _write_docker_compose(self, path: str):
        """Generate docker-compose.yml."""
        db_type = self.config.get('database', 'postgresql')
        
        db_services = {
            'postgresql': '''
  db:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${{DB_USER:-postgres}}
      POSTGRES_PASSWORD: ${{DB_PASSWORD:-postgres}}
      POSTGRES_DB: ${{DB_NAME:-{name}}}
    ports:
      - "${{DB_PORT:-5432}}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5''',
            'mysql': '''
  db:
    image: mysql:8.0
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: ${{DB_PASSWORD:-root}}
      MYSQL_DATABASE: ${{DB_NAME:-{name}}}
    ports:
      - "${{DB_PORT:-3306}}:3306"
    volumes:
      - mysql_data:/var/lib/mysql''',
            'sqlite': '',
        }
        
        db_service = db_services.get(db_type, '').format(name=self.project_name)
        
        db_volumes = {
            'postgresql': '\n  postgres_data:',
            'mysql': '\n  mysql_data:',
            'sqlite': '',
        }
        
        cache_config = self.config.get('cache', 'none')
        
        redis_service = ''
        redis_volume = ''
        if cache_config == 'redis':
            redis_service = '''
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data'''
            redis_volume = '\n  redis_data:'
        
        depends = []
        if db_type != 'sqlite':
            depends.append('db')
        if cache_config == 'redis':
            depends.append('redis')
        
        depends_str = ''
        if depends:
            depends_str = '\n    depends_on:\n' + '\n'.join(
                f'      - {d}' for d in depends
            )
        
        content = f'''# === APIGen Generated Docker Compose ===
# Project: {self.project_name}

version: "3.8"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "${{APP_PORT:-8000}}:8000"
    env_file:
      - .env
    environment:
      - ENVIRONMENT=production{depends_str}
    volumes:
      - app_uploads:/app/uploads
{db_service}{redis_service}

volumes:
  app_uploads:{db_volumes.get(db_type, '')}{redis_volume}
'''
        with open(path, 'w') as f:
            f.write(content)
    
    def _write_dockerignore(self, path: str):
        """Generate .dockerignore."""
        content = '''# === APIGen Generated .dockerignore ===
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
.env.local
.venv
venv
env
*.egg-info
dist
build
.pytest_cache
.mypy_cache
.coverage
htmlcov
node_modules
.DS_Store
Thumbs.db
*.md
docs/
tests/
docker-compose*.yml
Makefile
'''
        with open(path, 'w') as f:
            f.write(content)
    
    def _write_deploy_scripts(self, output_dir: str):
        """Generate deployment helper scripts."""
        scripts_dir = os.path.join(output_dir, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)
        
        # start.sh
        start_script = f'''#!/bin/bash
# === APIGen Deploy Script ===
# Project: {self.project_name}

set -e

echo "🚀 Starting {self.project_name}..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your settings"
fi

# Build and start
docker compose up --build -d

echo "✅ {self.project_name} is running!"
echo "📍 API: http://localhost:${{APP_PORT:-8000}}"
echo "📖 Docs: http://localhost:${{APP_PORT:-8000}}/docs"
'''
        start_path = os.path.join(scripts_dir, 'start.sh')
        with open(start_path, 'w') as f:
            f.write(start_script)
        os.chmod(start_path, 0o755)
        
        # stop.sh
        stop_script = f'''#!/bin/bash
echo "⏹️  Stopping {self.project_name}..."
docker compose down
echo "✅ Stopped."
'''
        stop_path = os.path.join(scripts_dir, 'stop.sh')
        with open(stop_path, 'w') as f:
            f.write(stop_script)
        os.chmod(stop_path, 0o755)


# ============================================================
# OPENAPI SPEC EXPORTER
# ============================================================

class OpenAPIExporter(BaseExporter):
    """Export project as OpenAPI specification."""
    
    def get_format_name(self) -> str:
        return "OpenAPI 3.0 Specification"
    
    def export(self, output_path: str = None) -> str:
        """Generate OpenAPI spec from project config."""
        if not output_path:
            output_path = f"{self.project_name}_openapi.json"
        
        spec = self._build_spec()
        
        self._ensure_output_dir(output_path)
        
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            content = self._to_yaml(spec)
            with open(output_path, 'w') as f:
                f.write(content)
        else:
            with open(output_path, 'w') as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)
        
        return os.path.abspath(output_path)
    
    def _build_spec(self) -> Dict[str, Any]:
        """Build OpenAPI 3.0 specification."""
        models = self.config.get('models', [])
        
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": f"{self.project_name} API",
                "description": f"Auto-generated API specification for {self.project_name}",
                "version": self.config.get('version', '1.0.0'),
                "contact": {
                    "name": "APIGen",
                    "url": "https://github.com/your-org/apigen"
                }
            },
            "servers": [
                {
                    "url": f"http://localhost:{self.config.get('port', 8000)}",
                    "description": "Development server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {}
            },
            "tags": []
        }
        
        # Add auth security scheme
        auth_type = self.config.get('auth', 'jwt')
        if auth_type == 'jwt':
            spec["components"]["securitySchemes"]["bearerAuth"] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        elif auth_type == 'api_key':
            spec["components"]["securitySchemes"]["apiKeyAuth"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }
        
        # Generate paths and schemas for each model
        for model in models:
            if isinstance(model, dict):
                model_name = model.get('name', '')
                fields = model.get('fields', [])
            elif hasattr(model, 'name'):
                model_name = model.name
                fields = model.fields if hasattr(model, 'fields') else []
            else:
                continue
            
            if not model_name:
                continue
            
            lower_name = model_name.lower()
            plural_name = self._pluralize(lower_name)
            
            # Add tag
            spec["tags"].append({
                "name": model_name,
                "description": f"Operations for {model_name}"
            })
            
            # Build schema
            schema = self._build_model_schema(model_name, fields)
            spec["components"]["schemas"][model_name] = schema
            spec["components"]["schemas"][f"{model_name}Create"] = {
                **schema,
                "description": f"Schema for creating a {model_name}"
            }
            
            # Build CRUD paths
            base_path = f"/api/{plural_name}"
            
            # GET list + POST
            spec["paths"][base_path] = {
                "get": {
                    "tags": [model_name],
                    "summary": f"List all {plural_name}",
                    "operationId": f"list_{plural_name}",
                    "parameters": [
                        {"name": "skip", "in": "query", "schema": {"type": "integer", "default": 0}},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 100}},
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": f"#/components/schemas/{model_name}"}
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "tags": [model_name],
                    "summary": f"Create a new {lower_name}",
                    "operationId": f"create_{lower_name}",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{model_name}Create"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": f"#/components/schemas/{model_name}"}
                                }
                            }
                        },
                        "422": {"description": "Validation error"}
                    }
                }
            }
            
            # GET one + PUT + DELETE
            item_path = f"{base_path}/{{id}}"
            spec["paths"][item_path] = {
                "get": {
                    "tags": [model_name],
                    "summary": f"Get {lower_name} by ID",
                    "operationId": f"get_{lower_name}",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True,
                         "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": f"#/components/schemas/{model_name}"}
                                }
                            }
                        },
                        "404": {"description": "Not found"}
                    }
                },
                "put": {
                    "tags": [model_name],
                    "summary": f"Update {lower_name}",
                    "operationId": f"update_{lower_name}",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True,
                         "schema": {"type": "integer"}}
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{model_name}Create"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Updated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": f"#/components/schemas/{model_name}"}
                                }
                            }
                        },
                        "404": {"description": "Not found"}
                    }
                },
                "delete": {
                    "tags": [model_name],
                    "summary": f"Delete {lower_name}",
                    "operationId": f"delete_{lower_name}",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True,
                         "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "204": {"description": "Deleted successfully"},
                        "404": {"description": "Not found"}
                    }
                }
            }
        
        return spec
    
    def _build_model_schema(self, model_name: str, 
                            fields: list) -> Dict[str, Any]:
        """Build JSON Schema for a model."""
        properties = {
            "id": {"type": "integer", "description": "Unique identifier"}
        }
        required = []
        
        type_mapping = {
            'string': {'type': 'string'},
            'text': {'type': 'string'},
            'integer': {'type': 'integer'},
            'float': {'type': 'number', 'format': 'float'},
            'boolean': {'type': 'boolean'},
            'datetime': {'type': 'string', 'format': 'date-time'},
            'date': {'type': 'string', 'format': 'date'},
            'email': {'type': 'string', 'format': 'email'},
            'url': {'type': 'string', 'format': 'uri'},
            'uuid': {'type': 'string', 'format': 'uuid'},
            'json': {'type': 'object'},
            'password': {'type': 'string', 'format': 'password'},
            'phone': {'type': 'string'},
            'slug': {'type': 'string', 'pattern': '^[a-z0-9-]+$'},
            'ip_address': {'type': 'string', 'format': 'ipv4'},
        }
        
        for f in fields:
            if isinstance(f, dict):
                fname = f.get('name', '')
                ftype = f.get('type', f.get('field_type', 'string'))
                is_required = f.get('required', not f.get('nullable', True))
            elif hasattr(f, 'name'):
                fname = f.name
                ftype = getattr(f, 'field_type', 'string')
                if hasattr(ftype, 'value'):
                    ftype = ftype.value
                is_required = getattr(f, 'required', False)
            else:
                continue
            
            if not fname:
                continue
            
            prop = type_mapping.get(str(ftype).lower(), {'type': 'string'}).copy()
            
            # Add constraints
            if isinstance(f, dict):
                if f.get('max_length'):
                    prop['maxLength'] = f['max_length']
                if f.get('min_length'):
                    prop['minLength'] = f['min_length']
                if f.get('enum_values'):
                    prop['enum'] = f['enum_values']
                if f.get('description'):
                    prop['description'] = f['description']
                if f.get('default') is not None:
                    prop['default'] = f['default']
            
            properties[fname] = prop
            
            if is_required:
                required.append(fname)
        
        schema = {
            "type": "object",
            "description": f"{model_name} model",
            "properties": properties,
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    @staticmethod
    def _pluralize(name: str) -> str:
        """Simple pluralization."""
        if name.endswith('y') and name[-2:] not in ('ay', 'ey', 'oy', 'uy'):
            return name[:-1] + 'ies'
        elif name.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return name + 'es'
        return name + 's'
    
    @staticmethod
    def _to_yaml(data: Dict, indent: int = 0) -> str:
        """Simple dict to YAML converter (basic, no dependency)."""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(OpenAPIExporter._to_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        for k, v in item.items():
                            lines.append(f"{prefix}    {k}: {v}")
                    else:
                        lines.append(f"{prefix}  - {item}")
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
            elif isinstance(value, str):
                if '\n' in value or ':' in value or '#' in value:
                    lines.append(f'{prefix}{key}: "{value}"')
                else:
                    lines.append(f"{prefix}{key}: {value}")
            elif value is None:
                lines.append(f"{prefix}{key}: null")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)


# ============================================================
# PROJECT REPORT EXPORTER
# ============================================================

class ReportExporter(BaseExporter):
    """Export project analysis report."""
    
    def get_format_name(self) -> str:
        return "Project Report"
    
    def export(self, output_path: str = None) -> str:
        """Generate project analysis report."""
        if not output_path:
            output_path = f"{self.project_name}_report.md"
        
        report = self._build_report()
        
        self._ensure_output_dir(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return os.path.abspath(output_path)
    
    def _build_report(self) -> str:
        """Build markdown report."""
        files = self._collect_files()
        
        # Calculate statistics
        total_lines = 0
        total_size = 0
        file_stats = []
        
        for fp in files:
            size = os.path.getsize(fp)
            total_size += size
            
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
            except (UnicodeDecodeError, IsADirectoryError):
                lines = 0
            
            total_lines += lines
            file_stats.append({
                'path': self._get_relative_path(fp),
                'lines': lines,
                'size': size,
            })
        
        file_stats.sort(key=lambda x: x['lines'], reverse=True)
        
        # Count by extension
        ext_counts: Dict[str, int] = {}
        for fs in file_stats:
            ext = os.path.splitext(fs['path'])[1] or '(none)'
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        # Build models info
        models = self.config.get('models', [])
        model_names = []
        for m in models:
            if isinstance(m, dict):
                model_names.append(m.get('name', 'Unknown'))
            elif hasattr(m, 'name'):
                model_names.append(m.name)
        
        # Format report
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 📊 Project Report: {self.project_name}

> Generated by APIGen on {now}

## Overview

| Metric | Value |
|--------|-------|
| Project Name | {self.project_name} |
| Database | {self.config.get('database', 'N/A')} |
| Auth Type | {self.config.get('auth', 'N/A')} |
| Total Files | {len(files)} |
| Total Lines | {total_lines:,} |
| Total Size | {self._format_size(total_size)} |
| Models | {len(model_names)} |

## Models

{chr(10).join(f'- **{name}**' for name in model_names) if model_names else '- No models defined'}

## File Breakdown

| File | Lines | Size |
|------|-------|------|
{chr(10).join(f"| {fs['path']} | {fs['lines']:,} | {self._format_size(fs['size'])} |" for fs in file_stats[:20])}

## File Types

| Extension | Count |
|-----------|-------|
{chr(10).join(f"| {ext} | {count} |" for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True))}

## API Endpoints (Estimated)

{chr(10).join(f"- `GET    /api/{self._pluralize(n.lower())}` — List all" + chr(10) + f"- `POST   /api/{self._pluralize(n.lower())}` — Create new" + chr(10) + f"- `GET    /api/{self._pluralize(n.lower())}/{{id}}` — Get by ID" + chr(10) + f"- `PUT    /api/{self._pluralize(n.lower())}/{{id}}` — Update" + chr(10) + f"- `DELETE /api/{self._pluralize(n.lower())}/{{id}}` — Delete" for n in model_names) if model_names else '- No endpoints'}

---
*Report generated by [APIGen](https://github.com/your-org/apigen)*
"""
        return report
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def _pluralize(name: str) -> str:
        """Simple pluralize."""
        if name.endswith('y') and name[-2:] not in ('ay', 'ey', 'oy', 'uy'):
            return name[:-1] + 'ies'
        elif name.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return name + 'es'
        return name + 's'


# ============================================================
# REQUIREMENTS EXPORTER
# ============================================================

class RequirementsExporter(BaseExporter):
    """Export different requirement file formats."""
    
    def get_format_name(self) -> str:
        return "Requirements"
    
    # Base dependencies always needed
    BASE_DEPS = {
        'fastapi': '>=0.104.0',
        'uvicorn[standard]': '>=0.24.0',
        'pydantic[email]': '>=2.5.0',
        'pydantic-settings': '>=2.1.0',
        'python-multipart': '>=0.0.6',
    }
    
    # Database drivers
    DB_DEPS = {
        'postgresql': {'asyncpg': '>=0.29.0', 'psycopg2-binary': '>=2.9.9'},
        'mysql': {'aiomysql': '>=0.2.0', 'mysqlclient': '>=2.2.0'},
        'sqlite': {'aiosqlite': '>=0.19.0'},
        'mongodb': {'motor': '>=3.3.0', 'pymongo': '>=4.6.0'},
    }
    
    # ORM
    ORM_DEPS = {
        'sqlalchemy': '>=2.0.23',
        'alembic': '>=1.13.0',
    }
    
    # Auth dependencies
    AUTH_DEPS = {
        'jwt': {
            'python-jose[cryptography]': '>=3.3.0',
            'passlib[bcrypt]': '>=1.7.4',
        },
        'oauth2': {
            'authlib': '>=1.2.0',
            'httpx': '>=0.25.0',
        },
        'session': {
            'itsdangerous': '>=2.1.0',
        },
    }
    
    # Optional features
    OPTIONAL_DEPS = {
        'redis': {'redis': '>=5.0.0', 'aioredis': '>=2.0.0'},
        'celery': {'celery': '>=5.3.0'},
        'cors': {},  # Built into FastAPI
        'testing': {
            'pytest': '>=7.4.0',
            'pytest-asyncio': '>=0.23.0',
            'httpx': '>=0.25.0',
            'factory-boy': '>=3.3.0',
        },
        'monitoring': {
            'prometheus-client': '>=0.19.0',
            'sentry-sdk[fastapi]': '>=1.38.0',
        },
    }
    
    def export(self, output_path: str = None) -> str:
        """Generate requirements.txt."""
        if not output_path:
            output_path = os.path.join(self.project_dir, 'requirements.txt')
        
        deps = self._collect_dependencies()
        
        lines = [
            f"# === {self.project_name} Dependencies ===",
            f"# Generated by APIGen on {datetime.datetime.now().strftime('%Y-%m-%d')}",
            "",
            "# --- Core ---",
        ]
        
        for pkg, ver in self.BASE_DEPS.items():
            lines.append(f"{pkg}{ver}")
        
        lines.append("\n# --- Database ---")
        db_type = self.config.get('database', 'postgresql')
        for pkg, ver in self.DB_DEPS.get(db_type, {}).items():
            lines.append(f"{pkg}{ver}")
        
        if db_type != 'mongodb':
            lines.append("\n# --- ORM ---")
            for pkg, ver in self.ORM_DEPS.items():
                lines.append(f"{pkg}{ver}")
        
        auth_type = self.config.get('auth', 'jwt')
        auth_deps = self.AUTH_DEPS.get(auth_type, {})
        if auth_deps:
            lines.append(f"\n# --- Auth ({auth_type}) ---")
            for pkg, ver in auth_deps.items():
                lines.append(f"{pkg}{ver}")
        
        # Optional deps
        features = self.config.get('features', [])
        for feature in features:
            feature_deps = self.OPTIONAL_DEPS.get(feature, {})
            if feature_deps:
                lines.append(f"\n# --- {feature.title()} ---")
                for pkg, ver in feature_deps.items():
                    lines.append(f"{pkg}{ver}")
        
        lines.append("")
        
        self._ensure_output_dir(output_path)
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return os.path.abspath(output_path)
    
    def _collect_dependencies(self) -> Dict[str, str]:
        """Collect all required dependencies."""
        deps = dict(self.BASE_DEPS)
        
        db_type = self.config.get('database', 'postgresql')
        deps.update(self.DB_DEPS.get(db_type, {}))
        
        if db_type != 'mongodb':
            deps.update(self.ORM_DEPS)
        
        auth_type = self.config.get('auth', 'jwt')
        deps.update(self.AUTH_DEPS.get(auth_type, {}))
        
        return deps
    
    def export_dev(self, output_path: str = None) -> str:
        """Generate requirements-dev.txt."""
        if not output_path:
            output_path = os.path.join(self.project_dir, 'requirements-dev.txt')
        
        lines = [
            f"# === {self.project_name} Dev Dependencies ===",
            "-r requirements.txt",
            "",
            "# --- Testing ---",
        ]
        
        for pkg, ver in self.OPTIONAL_DEPS['testing'].items():
            lines.append(f"{pkg}{ver}")
        
        lines.extend([
            "",
            "# --- Code Quality ---",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "",
            "# --- Development ---",
            "ipython>=8.18.0",
            "rich>=13.7.0",
            "pre-commit>=3.6.0",
            "",
        ])
        
        self._ensure_output_dir(output_path)
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return os.path.abspath(output_path)


# ============================================================
# EXPORTER FACTORY
# ============================================================

class ExporterFactory:
    """Factory for creating exporters."""
    
    _exporters = {
        'zip': ZipExporter,
        'docker': DockerExporter,
        'openapi': OpenAPIExporter,
        'report': ReportExporter,
        'requirements': RequirementsExporter,
    }
    
    @classmethod
    def create(cls, format_name: str, project_dir: str,
               config: Dict[str, Any] = None) -> BaseExporter:
        """Create an exporter by format name."""
        exporter_class = cls._exporters.get(format_name.lower())
        
        if not exporter_class:
            available = ', '.join(cls._exporters.keys())
            raise ValueError(
                f"Unknown export format '{format_name}'. "
                f"Available: {available}"
            )
        
        return exporter_class(project_dir, config)
    
    @classmethod
    def available_formats(cls) -> List[str]:
        """List available export formats."""
        return list(cls._exporters.keys())
    
    @classmethod
    def register(cls, name: str, exporter_class: type):
        """Register a custom exporter."""
        if not issubclass(exporter_class, BaseExporter):
            raise TypeError("Exporter must inherit from BaseExporter")
        cls._exporters[name.lower()] = exporter_class
    
    @classmethod
    def export_all(cls, project_dir: str, config: Dict[str, Any],
                   output_dir: str = None) -> Dict[str, str]:
        """Export in all available formats."""
        results = {}
        base_dir = output_dir or os.path.dirname(project_dir)
        
        for fmt in cls._exporters:
            try:
                exporter = cls.create(fmt, project_dir, config)
                output = exporter.export(
                    os.path.join(base_dir, f"export_{fmt}")
                )
                results[fmt] = output
            except Exception as e:
                results[fmt] = f"ERROR: {str(e)}"
        
        return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def export_to_zip(project_dir: str, output_path: str = None,
                  config: Dict = None) -> str:
    """Quick export to ZIP."""
    return ZipExporter(project_dir, config or {}).export(output_path)


def export_to_docker(project_dir: str, output_path: str = None,
                     config: Dict = None) -> str:
    """Quick export with Docker setup."""
    return DockerExporter(project_dir, config or {}).export(output_path)


def export_openapi(project_dir: str, output_path: str = None,
                   config: Dict = None) -> str:
    """Quick export OpenAPI spec."""
    return OpenAPIExporter(project_dir, config or {}).export(output_path)


def generate_report(project_dir: str, output_path: str = None,
                    config: Dict = None) -> str:
    """Quick generate project report."""
    return ReportExporter(project_dir, config or {}).export(output_path)


def generate_requirements(project_dir: str, config: Dict = None) -> str:
    """Quick generate requirements.txt."""
    return RequirementsExporter(project_dir, config or {}).export()
