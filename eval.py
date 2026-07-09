"""Evaluate every model in ALL_MODELS with 5-fold stratified CV.

Selection step (not deployment training). Use train.py to persist the
final chosen model.

ClassifierResult bundles all the numbers a downstream reporter/plot needs
per model: per-fold accuracy, OOF confusion matrix, sklearn classification
report string, and (if the estimator exposes them) averaged feature
importances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gesture import (
    ALL_MODELS,
    Classifier,
    GestureDataset,
    get_logger,
)


_log = get_logger("eval")


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
    model: Classifier, dataset: GestureDataset,
    n_splits: int = 5, seed: int = 0,
) -> ClassifierResult:
    """5-fold stratified CV on a Pipeline(StandardScaler, model.build()).

    Concatenates out-of-fold predictions to build one confusion matrix and
    one classification report over all samples. Feature importances (if the
    estimator exposes them) are averaged across folds.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
    )

    X = dataset.X
    y = dataset.y
    labels = dataset.labels

    # sklearn's MLP + early_stopping does not accept string labels
    # (it calls np.isnan on y_pred internally). Fit once, use int labels
    # for training, and map back to strings for reporting.
    y_int, le = dataset.y_int()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs: list[float] = []
    oof_true: list[str] = []
    oof_pred: list[str] = []
    importances_by_fold: list[np.ndarray] = []

    for tr_idx, te_idx in skf.split(X, y_int):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model.build(seed=seed)),
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
        name=model.name,
        accuracies=fold_accs,
        classification_report=report,
        confusion_matrix=cm,
        labels=labels,
        feature_importances=importances,
        feature_names=list(dataset.feature_names),
    )


def evaluate_all(
    dataset: GestureDataset,
    seed: int = 0,
    models: list[Classifier] | None = None,
) -> dict[str, ClassifierResult]:
    """Run the given models through 5-fold CV and collect results.

    `models` defaults to ALL_MODELS when omitted.
    """
    results: dict[str, ClassifierResult] = {}
    for model in (models if models is not None else ALL_MODELS):
        _log.info("evaluating %s...", model.name)
        results[model.name] = evaluate(model, dataset, seed=seed)
    return results
