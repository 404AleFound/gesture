"""Text reports for a batch of ClassifierResult objects.

Uses the project's logger so output automatically fans out to both stderr
and the timestamped log file installed by install_run_logger().

Each function builds a single multi-line block and emits it in one
`logger.info()` call so log-file formatting stays aligned (a per-line
call would prepend the timestamp / level to every row and break table
readability).
"""

from __future__ import annotations

from gesture import get_logger

from eval import ClassifierResult


_log = get_logger("report")


def log_leaderboard(results: dict[str, ClassifierResult]) -> None:
    """Sorted table of mean accuracy across classifiers."""
    rows = sorted(results.values(), key=lambda r: r.acc_mean, reverse=True)
    name_w = max(len(r.name) for r in rows) + 2
    sep = "=" * (name_w + 30)
    lines = [
        "",
        sep,
        "classifier".ljust(name_w) + "acc_mean  acc_std  folds",
        "-" * (name_w + 30),
    ]
    for r in rows:
        folds_str = " ".join(f"{a:.3f}" for a in r.accuracies)
        lines.append(
            f"{r.name.ljust(name_w)}{r.acc_mean:.4f}    {r.acc_std:.4f}  [{folds_str}]"
        )
    lines.append(sep)
    _log.info("\n".join(lines))


def log_full_reports(results: dict[str, ClassifierResult]) -> None:
    """Log sklearn's per-class classification_report for every model."""
    for r in results.values():
        block = [
            "",
            f"### {r.name}  (acc = {r.acc_mean:.4f} ± {r.acc_std:.4f})",
            r.classification_report.rstrip(),
        ]
        _log.info("\n".join(block))
