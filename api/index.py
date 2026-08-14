"""Vercel serverless entry point.

Vercel's Python runtime looks for an ASGI application named `app` in this file.
Everything real lives in the `app` package; this exists so the platform has the
single module path it expects, and so the deployment target is obvious to a
reader rather than buried in configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The function's working directory is not the repo root, so the package the app
# lives in has to be put on the path explicitly before it can be imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.main import app  # noqa: E402

__all__ = ["app"]
