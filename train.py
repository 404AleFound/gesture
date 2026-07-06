"""Train every model on the full dataset and persist to disk.

Cross-validation lives in eval.py (that's the *selection* step).
Once you know which model to deploy, use this script to train on 100% of
the data and save the fitted Pipeline (scaler + estimator) to a .joblib
file so it can be reloaded for inference without re-training.

CLI:
    python train.py data/gestures-....jsonl [--out-dir logger/models/]

Output layout:
    <out_dir>/<ModelName>.joblib     -- one per Classifier in ALL_MODELS
    <out_dir>/metadata.json          -- feature schema + provenance
    logger/train-<timestamp>.txt     -- run log (leaderboard-free)
    logger/latest.txt                -- most recent run's log
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture import (
    ALL_MODELS,
    Classifier,
    GestureDataset,
    current_run_dir,
    finalize_run_logger,
    get_logger,
    install_run_logger,
)


_log = get_logger("train")


def train_one(
    model: Classifier, dataset: GestureDataset, seed: int = 0,
) -> Pipeline:
    """Fit a StandardScaler + estimator pipeline on the full dataset.

    Uses integer-encoded labels to keep sklearn's internal checks (esp.
    MLP's np.isnan on y_pred) happy. The Pipeline stores the LabelEncoder
    is NOT persisted here; the caller writes it into metadata separately
    so the same encoder can be reused at inference time.
    """
    y_int, _ = dataset.y_int()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model.build(seed=seed)),
    ])
    pipe.fit(dataset.X, y_int)
    return pipe


def train_all(
    dataset: GestureDataset,
    out_dir: str | Path = "models",
    seed: int = 0,
    source_jsonl: str | Path | None = None,
) -> dict[str, Path]:
    """Train every model in ALL_MODELS and dump each to <out_dir>/<name>.joblib.

    Returns a mapping from model name to the joblib file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, le = dataset.y_int()
    label_order = list(le.classes_)

    paths: dict[str, Path] = {}
    for model in ALL_MODELS:
        _log.info("training %s...", model.name)
        pipe = train_one(model, dataset, seed=seed)
        path = out_dir / f"{model.name}.joblib"
        # Persist the pipeline together with the label decoder so a caller
        # only needs a single file to run inference end-to-end.
        joblib.dump(
            {"pipeline": pipe, "labels": label_order,
             "feature_names": list(dataset.feature_names)},
            path,
        )
        paths[model.name] = path

    metadata: dict[str, Any] = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": str(source_jsonl) if source_jsonl else None,
        "n_samples": len(dataset),
        "n_features": dataset.n_features,
        "feature_names": list(dataset.feature_names),
        "labels": label_order,
        "seed": seed,
        "models": {name: str(p.name) for name, p in paths.items()},
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return paths


def load_model(path: str | Path) -> tuple[Pipeline, list[str], list[str]]:
    """Load a saved model bundle. Returns (pipeline, labels, feature_names).

    Predictions from the pipeline are integer class indices; use
    `labels[int(pred)]` to recover the original gesture string.
    """
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["labels"], bundle["feature_names"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train all models and save them.")
    p.add_argument("jsonl", help="path to gesture jsonl file")
    p.add_argument("--log-dir", default="logger",
                   help="log directory (default: logger/)")
    p.add_argument("--out-dir", default=None,
                   help="override where models are written "
                        "(default: <log_dir>/<timestamp>/models/)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_dir = install_run_logger("train", log_dir=args.log_dir)
    _log.info("source: %s", args.jsonl)
    _log.info("run dir: %s", run_dir)

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "models"

    dataset = GestureDataset.from_jsonl(args.jsonl)
    _log.info("loaded %r", dataset)
    paths = train_all(
        dataset,
        out_dir=out_dir,
        seed=args.seed,
        source_jsonl=args.jsonl,
    )
    _log.info("saved %d model(s) to %s/", len(paths), out_dir)
    for name, path in paths.items():
        _log.info("  %-14s -> %s", name, path)

    latest = finalize_run_logger(log_dir=args.log_dir)
    _log.info("run saved: %s (latest -> %s)", run_dir, latest)


if __name__ == "__main__":
    main()
