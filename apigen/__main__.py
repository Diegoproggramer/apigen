# File: apigen/__main__.py
"""
NexaFlow APIGen — Module entry point.

Allows running the generator directly via::

    python -m apigen --schema schema.yaml --output ./generated_api

This module simply delegates to the CLI entry point defined in ``apigen.cli``.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Delegate to the CLI main function."""
    from apigen.cli import cli_main
    cli_main()


if __name__ == "__main__":
    main()
