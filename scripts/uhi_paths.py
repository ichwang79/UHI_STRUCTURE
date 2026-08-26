"""uhi_paths.py — resolve a deposited filename across the two data deposits.

The analysis reads from two records: the air-temperature UHI dataset and its companion panels.
They may be unpacked into one directory or kept apart. Every script here resolves a filename the
same way — the air deposit first, then the companion — so either layout works.

    UHI_AIR_DATA       the air-temperature UHI data deposit
    UHI_AIR_COMPANION  the companion data deposit (defaults to UHI_AIR_DATA)

Not run directly.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["deposits", "find"]


def deposits(air: str | Path | None = None,
             companion: str | Path | None = None) -> tuple[Path, Path]:
    """The two deposit directories, from arguments then environment."""
    a = air or os.environ.get("UHI_AIR_DATA") or os.environ.get("UHI_P3_DIR") or ""
    c = companion or os.environ.get("UHI_AIR_COMPANION") or a
    return Path(a).expanduser(), Path(c).expanduser()


def find(name: str, air: str | Path | None = None,
         companion: str | Path | None = None) -> Path:
    """Locate one deposited file, air deposit first, then companion."""
    a, c = deposits(air, companion)
    for base in (a, c):
        if str(base) and (base / name).is_file():
            return base / name
    raise SystemExit(
        f"cannot find {name} in either data deposit.\n"
        f"  air deposit:       {a or '(unset)'}\n"
        f"  companion deposit: {c or '(unset)'}\n"
        "Set UHI_AIR_DATA and UHI_AIR_COMPANION, or pass the directories on the command line.\n"
        "Both may be the same directory if the deposits were unpacked together.")
