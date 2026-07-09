"""Project-wide logging setup.

Every action script (train / eval / plot) calls `install_run_logger()`
once in its main() to wire up:

    - a stderr handler (short single-line format, human-friendly)
    - a file handler under `<log_dir>/<timestamp>/<name>.log`

The whole run stores its artifacts in that timestamp directory:

    logger/
    └── 20260706-120400/
        ├── eval.log
        ├── models/          (train.py drops joblibs here)
        ├── kotlin/          (export.py drops .kt files here)
        └── figs/            (plot.py / eval.py write figures here)

There is no `latest` symlink -- each run stands on its own so a stale
export cannot silently pair up with a newer training run.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path


_STREAM_FMT = "%(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"

# Tracks the current run's directory so finalize_run_logger() and outside
# callers (e.g. train.py wanting to place models/) can locate it without
# having to pass the path around.
_current_run_dir: Path | None = None


def install_run_logger(
    name: str,
    log_dir: str | Path = "logger",
    level: int = logging.INFO,
) -> Path:
    """Configure root logger for a script run. Returns the run directory.

    Layout: `<log_dir>/<yyyymmdd-HHMMSS>/<name>.log`. The whole timestamp
    directory is created up front; other artifacts belonging to the same
    run (models, figures, ...) should be written under it via
    `current_run_dir()`.

    Any existing handlers on the root logger are removed first so
    re-invocation in the same process (tests, notebooks) doesn't double
    up output.
    """
    global _current_run_dir

    log_dir = Path(log_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"{name}.log"

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter(_STREAM_FMT))
    root.addHandler(stream)

    file = logging.FileHandler(log_file, encoding="utf-8")
    file.setLevel(level)
    file.setFormatter(logging.Formatter(_FILE_FMT))
    root.addHandler(file)

    _current_run_dir = run_dir
    return run_dir


def current_run_dir() -> Path:
    """Return the directory for the current run (must be after install_...)."""
    if _current_run_dir is None:
        raise RuntimeError(
            "install_run_logger() must be called before current_run_dir()"
        )
    return _current_run_dir


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper so callers don't have to import logging directly."""
    return logging.getLogger(name)
