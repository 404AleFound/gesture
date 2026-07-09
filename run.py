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

train.py still works standalone for training-only workflows; eval and
plot are only reachable through this script now.

CLI:
    python run.py data/gestures-....jsonl
        [--log-dir logger/] [--seed 0]
        [--models RandomForest MLP ...]

EXP:
python3 run.py data/gestures-20260703-*.jsonl --model RandomForest

Use `--models` to restrict which classifiers are trained/evaluated; the
default is every entry in gesture.ALL_MODELS.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from gesture import (
    ALL_MODELS,
    Classifier,
    FEATURE_NAMES,
    GestureDataset,
    get_logger,
    install_run_logger,
    load_scrolls,
)
from gesture.plots import plot_all, plot_classifier_results

from eval import ClassifierResult, evaluate_all
from train import train_all


_log = get_logger("run")


# ---------- schema-drift guard ----------


_KT_FEATURES_RE = re.compile(
    r'val\s+featureNames\s*:\s*List<String>\s*=\s*listOf\(([^)]*)\)'
)


def _kt_feature_names(kt_path: Path) -> list[str] | None:
    """Parse featureNames out of a generated Kotlin file, or None if absent."""
    try:
        m = _KT_FEATURES_RE.search(kt_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not m:
        return None
    return re.findall(r'"([^"]*)"', m.group(1))


def warn_stale_kotlin_exports(log_dir: Path) -> None:
    """Log any historical `.kt` exports whose feature schema differs from
    the current `FEATURE_NAMES` whitelist. The exports themselves are left
    on disk (each run stands on its own) -- the warning just makes it
    obvious which timestamps are no longer safe to ship."""
    current = list(FEATURE_NAMES)
    stale: list[tuple[Path, list[str]]] = []
    if log_dir.exists():
        for kt in sorted(log_dir.glob("*/kotlin/*.kt")):
            names = _kt_feature_names(kt)
            if names is not None and names != current:
                stale.append((kt, names))
    if not stale:
        return
    lines = ["", f"[schema drift] {len(stale)} kt export(s) use an older feature set:"]
    for path, names in stale:
        added = sorted(set(current) - set(names))
        removed = sorted(set(names) - set(current))
        detail = []
        if added:
            detail.append(f"+{added}")
        if removed:
            detail.append(f"-{removed}")
        lines.append(f"  {path}  ({', '.join(detail) or 'reordered'})")
    lines.append("These files will not match this run's models. "
                 "Re-export from a fresh training run if you plan to ship them.")
    _log.warning("\n".join(lines))


# ---------- model selection ----------


def _select_models(names: list[str] | None) -> list[Classifier]:
    """Filter ALL_MODELS by name; unknown names raise with the valid set."""
    if not names:
        return list(ALL_MODELS)
    by_name = {m.name: m for m in ALL_MODELS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(
            f"unknown model(s): {missing}. "
            f"valid: {sorted(by_name)}"
        )
    return [by_name[n] for n in names]


# ---------- reporting (previously report.py) ----------


def log_feature_schema(dataset: GestureDataset) -> None:
    """Emit the ordered list of features the models will see."""
    names = list(dataset.feature_names)
    width = max(len(n) for n in names)
    lines = ["", f"model input: {len(names)} feature(s)"]
    lines.extend(f"  [{i:>2}] {n.ljust(width)}" for i, n in enumerate(names))
    _log.info("\n".join(lines))


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


# ---------- main ----------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Run the full pipeline (train + eval + plot) into one "
                    "timestamp-stamped run directory.")
    p.add_argument("jsonl", help="path to gesture jsonl file")
    p.add_argument("--log-dir", default="logger",
                   help="log directory (default: logger/)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        default=None,
        help="subset of models to train/evaluate "
             f"(default: all — {', '.join(m.name for m in ALL_MODELS)})",
    )
    p.add_argument("--plot", action="store_true",
                   help="also render scroll/feature/distribution figures")
    p.add_argument("--label-mode", default="hand",
                   choices=["hand", "hand1", "gesture", "holdmode"],
                   help="target for classification (default: hand). "
                        "'holdmode' = THUMB(单手) vs INDEX(双手).")
    p.add_argument("--only-tags", nargs="+", default=None,
                   metavar="TAG",
                   choices=["LEFT_THUMB", "RIGHT_THUMB",
                            "LEFT_INDEX", "RIGHT_INDEX"],
                   help="restrict to scrolls with tag in this set (used for "
                        "cascade stages, e.g. train hand head only on thumb).")
    args = p.parse_args(argv)

    models = _select_models(args.models)

    run_dir = install_run_logger("run", log_dir=args.log_dir)
    _log.info("source: %s", args.jsonl)
    _log.info("run dir: %s", run_dir)
    _log.info("models: %s", [m.name for m in models])

    # ---- shared inputs ----
    _log.info("label mode: %s", args.label_mode)
    scrolls = list(load_scrolls(args.jsonl))
    if args.only_tags:
        kept = {t for t in args.only_tags}
        before = len(scrolls)
        scrolls = [s for s in scrolls if s.tag in kept]
        _log.info("only-tags filter %s: %d -> %d scrolls",
                  sorted(kept), before, len(scrolls))
    dataset = GestureDataset.from_scrolls(scrolls, label_mode=args.label_mode)
    _log.info("loaded %r", dataset)
    log_feature_schema(dataset)
    warn_stale_kotlin_exports(Path(args.log_dir))

    # ---- 1. train ----
    _log.info("=== phase: train ===")
    train_all(
        dataset,
        out_dir=run_dir / "models",
        seed=args.seed,
        source_jsonl=args.jsonl,
        models=models,
    )

    # ---- 2. evaluate ----
    _log.info("=== phase: eval ===")
    results = evaluate_all(dataset, seed=args.seed, models=models)
    log_leaderboard(results)
    log_full_reports(results)
    plot_classifier_results(results, out_dir=run_dir / "figs")

    # ---- 3. plot ----
    if args.plot:
        _log.info("=== phase: plot ===")
        plot_all(scrolls, out_root=run_dir / "plots")
        _log.info("figures written to %s/", run_dir / "plots")

    _log.info("run saved: %s", run_dir)


if __name__ == "__main__":
    main()
