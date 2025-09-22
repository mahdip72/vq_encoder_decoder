"""Minimal path constants required by the trimmed ProteinWorkshop fork."""

from pathlib import Path

SRC_PATH = Path(__file__).parent
PROJECT_PATH = SRC_PATH.parent

__all__ = ["SRC_PATH", "PROJECT_PATH"]
