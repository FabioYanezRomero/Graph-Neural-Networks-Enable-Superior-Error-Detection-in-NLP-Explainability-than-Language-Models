#!/usr/bin/env python3
"""Compatibility wrapper for legacy entry points.

This shim delegates to ``aggregate_consistency.py`` to preserve backwards
compatibility with older automation that still imports or executes
``aggregate_contrastivity.py`` directly.
"""

from .aggregate_consistency import main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    main()
