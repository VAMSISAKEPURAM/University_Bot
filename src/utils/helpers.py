import logging
import os
from pathlib import Path
import yaml


def load_config() -> dict:
    """
    Locate and load config.yaml.

    Search order (most-specific first):
      1. Project root  → <repo>/config.yaml          (when run via Streamlit / app.py)
      2. src parent    → <repo>/src/../config.yaml    (same as above, kept for safety)
      3. CWD           → ./config.yaml               (fallback for edge cases)
    """
    candidates = [
        # helpers.py lives at src/utils/helpers.py → go up 3 levels to project root
        Path(__file__).resolve().parents[2] / "config.yaml",
        # original path: src/../config.yaml  (2 levels up from helpers.py)
        Path(__file__).resolve().parents[1] / "config.yaml",
        # current working directory
        Path(os.getcwd()) / "config.yaml",
    ]

    for config_path in candidates:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as config_file:
                return yaml.safe_load(config_file)

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"config.yaml not found. Searched:\n  {searched}"
    )


def setup_logging(level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
