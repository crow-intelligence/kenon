"""Shared utilities for the Google and the Mind experiment scripts."""

import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_pickle(path: Path) -> Any:
    """Load a pickle file with a clear FileNotFoundError if missing."""
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_pickle(obj: Any, path: Path) -> None:
    """Save to pickle, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def require_file(path: Path, hint: str) -> None:
    """Raise SystemExit with a helpful message if path does not exist.

    Args:
        path: File path to check.
        hint: Name of the script that should have created this file.
    """
    if not path.exists():
        log("error", f"Missing required file: {path}")
        log(
            "error",
            f"Run first: uv run python experiments/google_and_the_mind/scripts/{hint}",
        )
        sys.exit(1)


def log(step: str, message: str) -> None:
    """Print a timestamped log line: [step] message."""
    ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
    print(f"{ts} [{step}] {message}")
