"""Feature dataset assembled from Scroll records.

GestureDataset turns a list[Scroll] into the (X, y) matrices that sklearn
classifiers consume. It's the single place that owns the "which columns
live in the feature vector" decision, so any classifier script can just:

    ds = GestureDataset.from_jsonl("data/gestures-....jsonl")
    clf.fit(ds.X, ds.y)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from gesture.scrolls import DIRECTIONS, GESTURES, Scroll, load_scrolls
from gesture.feature import compute


# ---------- label modes ----------

# Two ways to derive y from Scroll.tag:
#   "hand"    -> collapse LEFT_*/RIGHT_* to LEFT_HAND/RIGHT_HAND
#                (main classification task: which hand the user is using)
#   "gesture" -> keep all 4 raw tags (fine-grained gesture identity,
#                mostly useful for data exploration and feature plots)
HAND_LABELS: tuple[str, ...] = ("LEFT_HAND", "RIGHT_HAND")

_TAG_TO_HAND = {
    "LEFT_THUMB":  "LEFT_HAND",
    "LEFT_INDEX":  "LEFT_HAND",
    "RIGHT_THUMB": "RIGHT_HAND",
    "RIGHT_INDEX": "RIGHT_HAND",
}


def _label_of(scroll: Scroll, mode: str) -> str:
    if mode == "hand":
        return _TAG_TO_HAND[scroll.tag]
    if mode == "gesture":
        return scroll.tag
    raise ValueError(f"unknown label mode: {mode!r}")


# ---------- feature-vector schema ----------

# Continuous features taken straight off the Features dataclass, in the
# order they will appear in each training row.
FEATURE_NAMES: tuple[str, ...] = (
    "length",

    "disp_total_dx", 
    "disp_total_dy", 
    "disp_max_dx", 
    "disp_max_dy",

    "velocity_max", 
    "velocity_mean", 
    "velocity_std",

    "curvature_rmse", 
    "curvature_max", 
    "curvature_mean",
    "convex_orientation",
)


def _row_for(scroll: Scroll) -> list[float]:
    """Return one feature row for one scroll, aligned with FEATURE_NAMES."""
    f = compute(scroll)
    row: list[float] = [getattr(f, name) for name in FEATURE_NAMES]
    return row


# ---------- Dataset ----------


@dataclass
class GestureDataset:
    """Feature matrix + labels + metadata for a batch of scrolls.

    Attributes:
        X: shape (n_samples, n_features), float64.
        y: shape (n_samples,), string labels. The specific set depends on
            the `label_mode` passed to the constructor -- by default it's
            ("LEFT_HAND", "RIGHT_HAND") for the hand-recognition task.
        feature_names: column names for X, aligned with FEATURE_NAMES.
        labels: sorted list of unique classes; useful as the `labels=`
                argument to sklearn metrics for stable row/col order.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    labels: list[str] = field(default_factory=lambda: list(HAND_LABELS))

    # ---- constructors ----

    @classmethod
    def from_scrolls(
        cls,
        scrolls: Iterable[Scroll],
        label_mode: str = "hand",
    ) -> "GestureDataset":
        rows: list[list[float]] = []
        ys: list[str] = []
        for scroll in scrolls:
            rows.append(_row_for(scroll))
            ys.append(_label_of(scroll, label_mode))
        if not rows:
            raise ValueError("GestureDataset needs at least one scroll")
        labels = list(HAND_LABELS) if label_mode == "hand" else sorted(GESTURES)
        return cls(
            X=np.asarray(rows, dtype=float),
            y=np.asarray(ys),
            labels=labels,
        )

    @classmethod
    def from_jsonl(
        cls, path: str | Path, label_mode: str = "hand",
    ) -> "GestureDataset":
        return cls.from_scrolls(load_scrolls(path), label_mode=label_mode)

    # ---- convenience ----

    def __len__(self) -> int:
        return self.X.shape[0]

    def __repr__(self) -> str:
        counts = {lbl: int((self.y == lbl).sum()) for lbl in self.labels}
        return (
            f"GestureDataset(n={len(self)}, "
            f"n_features={self.X.shape[1]}, "
            f"class_counts={counts})"
        )

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def y_int(self):
        """Integer-encoded labels + the fitted LabelEncoder.

        sklearn's MLPClassifier with early_stopping doesn't accept string
        labels (internal np.isnan check). Callers that need to feed such
        estimators should use this instead of y.
        """
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.labels)
        return le.transform(self.y), le

    def stratified_split(
        self, test_size: float = 0.2, seed: int = 0
    ) -> tuple["GestureDataset", "GestureDataset"]:
        """Split into (train, test) preserving per-class proportions."""
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=seed,
            stratify=self.y,
        )
        def make(X, y):
            return GestureDataset(
                X=X, y=y,
                feature_names=list(self.feature_names),
                labels=list(self.labels),
            )
        return make(X_tr, y_tr), make(X_te, y_te)


