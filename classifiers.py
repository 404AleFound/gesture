"""Task 3: reproduce five classifiers from the paper on our gesture data.

Pipeline:
    1. build_dataset(scrolls) -> (X, y, feature_names)
    2. evaluate(clf, X, y, name)  -- 5-fold stratified CV
    3. evaluate_all(scrolls)      -- run all five, return results dict
    4. plotting + leaderboard helpers

Design notes:
    - Labels are the 4 gesture modes (LEFT_THUMB / RIGHT_THUMB / LEFT_INDEX /
      RIGHT_INDEX). Direction is not a label; instead direction_rad is turned
      into (sin, cos) features so all 4 directions live in the same feature
      space without an angle-wraparound problem.
    - Every classifier runs inside a Pipeline(StandardScaler, clf). Trees
      ignore scaling but the extra cost is negligible and keeps the code
      uniform.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scrolls import DIRECTIONS, GESTURES, Scroll, load_scrolls
from features import TrajectoryFeatures, compute


# ---------- dataset ----------


def build_dataset(
    scrolls: list[Scroll],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names).

    Feature vector layout (23 dims):
        - 15 base features from TrajectoryFeatures (direction_rad excluded)
        - 2 direction encodings: dir_sin, dir_cos (continuous)
        - 2 start-point coordinates: start_x, start_y  (adds absolute
          position info that the trajectory-relative features lose)
        - 4 direction one-hot flags: is_RIGHT / is_LEFT / is_DOWN / is_UP
          (redundant with sin/cos but trees split cleanly on 0/1)
    """
    base_names = [
        "length", "displacement", "duration_ms", "point_count",
        "velocity_max", "velocity_mean", "velocity_std",
        "total_dx", "total_dy", "max_dx", "max_dy",
        "rmse", "curvature_max", "curvature_mean", "convex_orientation",
    ]
    dir_flag_names = [f"is_{d}" for d in DIRECTIONS]
    feature_names = (
        base_names
        + ["dir_sin", "dir_cos"]
        + ["start_x", "start_y"]
        + dir_flag_names
    )

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    for scroll in scrolls:
        f = compute(scroll)
        row = [getattr(f, name) for name in base_names]
        row.append(math.sin(f.direction_rad))
        row.append(math.cos(f.direction_rad))
        if scroll.path:
            row.append(scroll.path[0].x)
            row.append(scroll.path[0].y)
        else:
            row.append(0.0)
            row.append(0.0)
        direction = scroll.direction
        for d in DIRECTIONS:
            row.append(1.0 if d == direction else 0.0)
        X_rows.append(row)
        y_rows.append(scroll.tag)

    return np.asarray(X_rows, dtype=float), np.asarray(y_rows), feature_names


# ---------- results container ----------


@dataclass
class ClassifierResult:
    name: str
    accuracies: list[float]                # per-fold accuracy
    classification_report: str             # sklearn text report on concat'd oof preds
    confusion_matrix: np.ndarray           # 4x4, rows=true, cols=pred
    labels: list[str]                      # gesture order for the matrix
    feature_importances: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    @property
    def acc_mean(self) -> float:
        return float(np.mean(self.accuracies))

    @property
    def acc_std(self) -> float:
        return float(np.std(self.accuracies))


# ---------- evaluation ----------


def evaluate(
    clf, X: np.ndarray, y: np.ndarray,
    name: str, feature_names: list[str],
    n_splits: int = 5, seed: int = 0,
) -> ClassifierResult:
    """5-fold stratified CV on a Pipeline(StandardScaler, clf).

    Concatenates out-of-fold predictions to build one confusion matrix and
    one classification report over all samples. Feature importances (if the
    estimator exposes them) are averaged across folds.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
    )

    labels = sorted(GESTURES)  # stable canonical label order
    # sklearn's MLP + early_stopping does not accept string labels
    # (it calls np.isnan on y_pred internally). Encode to int and
    # map back after prediction. Fitting a LabelEncoder with the fixed
    # `labels` order keeps every fold on the same int-to-name mapping.
    le = LabelEncoder()
    le.fit(labels)
    y_int = le.transform(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs: list[float] = []
    oof_true: list[str] = []
    oof_pred: list[str] = []
    importances_by_fold: list[np.ndarray] = []

    for tr_idx, te_idx in skf.split(X, y_int):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf.__class__(**clf.get_params())),
        ])
        pipe.fit(X[tr_idx], y_int[tr_idx])
        pred_int = pipe.predict(X[te_idx])
        pred = le.inverse_transform(pred_int)
        fold_accs.append(accuracy_score(y[te_idx], pred))
        oof_true.extend(y[te_idx].tolist())
        oof_pred.extend(pred.tolist())

        est = pipe.named_steps["clf"]
        if hasattr(est, "feature_importances_"):
            importances_by_fold.append(est.feature_importances_)

    cm = confusion_matrix(oof_true, oof_pred, labels=labels)
    report = classification_report(oof_true, oof_pred, labels=labels, digits=3)

    importances = None
    if importances_by_fold:
        importances = np.mean(np.vstack(importances_by_fold), axis=0)

    return ClassifierResult(
        name=name,
        accuracies=fold_accs,
        classification_report=report,
        confusion_matrix=cm,
        labels=labels,
        feature_importances=importances,
        feature_names=feature_names,
    )


# ---------- run all five ----------


def evaluate_all(
    scrolls: list[Scroll], seed: int = 0,
) -> dict[str, ClassifierResult]:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier

    X, y, feature_names = build_dataset(scrolls)

    configs = [
        ("DecisionTree",
         DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)),
        ("RandomForest",
         RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        ("NaiveBayes",
         GaussianNB()),
        ("MLP",
         MLPClassifier(hidden_layer_sizes=(64, 32),
                       max_iter=2000,
                       n_iter_no_change=30,
                       tol=1e-5,
                       random_state=seed)),
        ("kNN",
         KNeighborsClassifier(n_neighbors=5)),
    ]

    results: dict[str, ClassifierResult] = {}
    for name, clf in configs:
        print(f"evaluating {name}...", file=sys.stderr)
        results[name] = evaluate(clf, X, y, name, feature_names, seed=seed)
    return results


# ---------- reporting ----------


def print_leaderboard(results: dict[str, ClassifierResult]) -> None:
    """Sorted table of mean accuracy across classifiers."""
    rows = sorted(results.values(), key=lambda r: r.acc_mean, reverse=True)
    name_w = max(len(r.name) for r in rows) + 2
    print()
    print("=" * (name_w + 30))
    print("classifier".ljust(name_w) + "acc_mean  acc_std  folds")
    print("-" * (name_w + 30))
    for r in rows:
        folds_str = " ".join(f"{a:.3f}" for a in r.accuracies)
        print(f"{r.name.ljust(name_w)}{r.acc_mean:.4f}    {r.acc_std:.4f}  [{folds_str}]")
    print("=" * (name_w + 30))


def print_full_reports(results: dict[str, ClassifierResult]) -> None:
    for r in results.values():
        print()
        print(f"### {r.name}  (acc = {r.acc_mean:.4f} ± {r.acc_std:.4f})")
        print(r.classification_report)


def plot_confusion_matrices(
    results: dict[str, ClassifierResult],
    out_path: str | Path = "figs/classifiers/confusion_matrices.png",
) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(results.items())
    n = len(items)
    # Fixed 2x3 grid for up to 6 classifiers.
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))

    for ax, (name, r) in zip(axes.flat, items):
        cm = r.confusion_matrix
        # Row-normalize so each cell shows recall for that true class.
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{cm[i, j]}\n{cm_norm[i, j]*100:.0f}%"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=color)

        ax.set_xticks(range(len(r.labels)))
        ax.set_yticks(range(len(r.labels)))
        ax.set_xticklabels(r.labels, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(r.labels, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"{name}\nacc = {r.acc_mean:.3f}", fontsize=9)

    # Hide any leftover axes.
    for ax in axes.flat[len(items):]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                 label="row-normalized frequency")
    fig.suptitle("Confusion matrices (5-fold OOF)", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(
    results: dict[str, ClassifierResult],
    out_path: str | Path = "figs/classifiers/feature_importance.png",
) -> None:
    """Bar plot of feature importances for models that expose them."""
    import matplotlib.pyplot as plt

    have = [r for r in results.values() if r.feature_importances is not None]
    if not have:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(have)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)
    for ax, r in zip(axes[:, 0], have):
        order = np.argsort(r.feature_importances)[::-1]
        names = [r.feature_names[i] for i in order]
        vals = r.feature_importances[order]
        ax.bar(range(len(vals)), vals, color="#4c72b0")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{r.name} — feature importance", fontsize=10)
        ax.set_ylabel("importance")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- entry ----------


def run(jsonl_path: str | Path, out_dir: str | Path = "figs/classifiers") -> None:
    scrolls = load_scrolls(jsonl_path)
    print(f"loaded {len(scrolls)} scrolls", file=sys.stderr)
    results = evaluate_all(scrolls)
    print_leaderboard(results)
    print_full_reports(results)
    plot_confusion_matrices(results, Path(out_dir) / "confusion_matrices.png")
    plot_feature_importances(results, Path(out_dir) / "feature_importance.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python classifiers.py <file.jsonl>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
