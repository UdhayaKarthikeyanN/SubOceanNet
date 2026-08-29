"""Shared helpers for the isolated live-data fetch scripts.

This module runs inside .venv-live (see requirements.txt in this folder),
never inside the main app's .venv. It has no dependency on the rest of the
SubOceanNet codebase so it can be invoked as a standalone subprocess.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_dotenv_if_present() -> None:
    """Load a .env file from the project root, if one exists. Never prints
    or logs credential values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            f"missing required environment variable(s): {', '.join(missing)} "
            f"(set them in .env at the project root - see .env.example)"
        )
    return {n: os.environ[n] for n in names}


def emit_result(*, variable: str, provider: str, observation_time: str, path: str) -> None:
    """Print the single JSON line the caller (src/data/live/providers.py)
    parses from stdout. Must be the last line printed."""
    print(json.dumps({
        "variable": variable,
        "provider": provider,
        "observation_time": observation_time,
        "path": str(path),
    }))


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)
