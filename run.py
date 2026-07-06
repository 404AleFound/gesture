"""One-shot pipeline: train + evaluate + plot into a single run directory.

Runs the three phases sequentially in the same process so they share one
`install_run_logger()` call and therefore one `<log_dir>/<timestamp>/`
folder. Layout:

    logger/20260706-121500/
    ├── run.log
    ├── models/*.joblib + metadata.json
    ├── figs/                 (eval's classifier figures)
    │   ├── confusion_matrices.png
    │   └── feature_importance.png
    └── plots/                (scroll-based figures)
        ├── scrolls/
        ├── features/
        └── distribution/

The individual entry points (train.py, eval.py, plot.py) still work
standalone; each creates its own run directory in that case.

CLI:
    python run.py data/gestures-....jsonl [--log-dir logger/] [--seed 0]
"""

from __future__ import annotations

import argparse

from gesture import (
    GestureDataset,
    finalize_run_logger,
    get_logger,
    install_run_logger,
    load_scrolls,
)
from gesture.plots import plot_all

from eval import evaluate_and_report
from train import train_all


_log = get_logger("run")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Run the full pipeline (train + eval + plot) into one "
                    "timestamp-stamped run directory.")
    p.add_argument("jsonl", help="path to gesture jsonl file")
    p.add_argument("--log-dir", default="logger",
                   help="log directory (default: logger/)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    run_dir = install_run_logger("run", log_dir=args.log_dir)
    _log.info("source: %s", args.jsonl)
    _log.info("run dir: %s", run_dir)

    # ---- shared inputs ----
    dataset = GestureDataset.from_jsonl(args.jsonl, label_mode="gesture")
    _log.info("loaded %r", dataset)
    scrolls = load_scrolls(args.jsonl)

    # ---- 1. train ----
    _log.info("=== phase: train ===")
    train_all(
        dataset,
        out_dir=run_dir / "models",
        seed=args.seed,
        source_jsonl=args.jsonl,
    )

    # ---- 2. evaluate ----
    _log.info("=== phase: eval ===")
    evaluate_and_report(dataset, figs_dir=run_dir / "figs", seed=args.seed)

    # ---- 3. plot ----
    _log.info("=== phase: plot ===")
    plot_all(scrolls, out_root=run_dir / "plots")
    _log.info("figures written to %s/", run_dir / "plots")

    latest = finalize_run_logger(log_dir=args.log_dir)
    _log.info("run saved: %s (latest -> %s)", run_dir, latest)


if __name__ == "__main__":
    main()
