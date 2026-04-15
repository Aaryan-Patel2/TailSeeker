"""Pytest configuration — add repo root to sys.path for src imports."""

import sys
from pathlib import Path

# Add parent directory (repo root) to path so src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
