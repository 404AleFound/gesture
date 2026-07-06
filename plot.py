"""CLI entry point for generating every scroll-based figure.

Figures land under `<log_dir>/<timestamp>/figs/` alongside the plot log,
matching the layout used by train.py and eval.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gesture import (
    finalize_run_logger,
    get_logger,
    install_run_logger,
    load_scrolls,
)
from gesture.plots import plot_all


_log = get_logger("plot")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Render every scroll-based figure.")
    p.add_argument("jsonl", help="path to gesture jsonl file")
    p.add_argument("--log-dir", default="logger",
                   help="log directory (default: logger/)")
    p.add_argument("--out-dir", default=None,
                   help="override where figures are written "
                        "(default: <log_dir>/<timestamp>/figs/)")
    args = p.parse_args(argv)

    run_dir = install_run_logger("plot", log_dir=args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figs"

    _log.info("source: %s", args.jsonl)
    _log.info("run dir: %s", run_dir)

    scrolls = load_scrolls(args.jsonl)
    _log.info("loaded %d scrolls", len(scrolls))

    plot_all(scrolls, out_root=out_dir)
    _log.info("figures written to %s/", out_dir)

    latest = finalize_run_logger(log_dir=args.log_dir)
    _log.info("run saved: %s (latest -> %s)", run_dir, latest)


if __name__ == "__main__":
    main()
