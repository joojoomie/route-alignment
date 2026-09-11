#!/usr/bin/env python3
"""Small shared I/O helpers: atomic writes and CSV rows.

Every artifact in this repository is written atomically: the writer receives a
temporary path in the target directory and the file is renamed into place only
after the writer returns, so a crash or an interrupt can never leave a
half-written mapping, manifest or seal that a later stage would silently read.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence


def atomic_write(target: Path, writer: Callable[[Path], None]) -> None:
    """Call `writer(temporary_path)` and rename the result onto `target`."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.stem}-", suffix=target.suffix, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Write dict rows with a header; combine with `atomic_write` for artifacts."""

    with Path(path).open("w", newline="") as handle:
        output = csv.DictWriter(handle, fieldnames=list(fieldnames))
        output.writeheader()
        output.writerows(rows)
