from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None, target_folder: str = "hest1k-dataset") -> Path:
    path = (start or Path.cwd()).resolve()

    for candidate in (path, *path.parents):
        if (candidate / target_folder).exists():
            return candidate

    raise FileNotFoundError(f"Could not find project root containing '{target_folder}' from {path}")


PROJECT_ROOT = find_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data300" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "phase1_baseline"
