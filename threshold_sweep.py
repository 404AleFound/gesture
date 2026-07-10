"""Threshold sweep for RandomForest so per-class precision >= 0.99.

Runs 5-fold stratified CV on the merged dataset, collects out-of-fold
probabilities, then for each class finds the smallest threshold t such
that predicting only when P(c) >= t yields precision >= 0.99. Reports
precision / recall / coverage at that threshold.

Coverage here = fraction of *all* samples of that class that the model
still chooses to label (i.e. abstains on the rest). Ship-relevant number.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture import GestureDataset

TARGET_PRECISION = 0.99
JSONL = "data/gestures-all.jsonl"

ds = GestureDataset.from_jsonl(JSONL, label_mode="hand")
X, y = ds.X, ds.y
y_int, le = None, None
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(y)
y_int = le.transform(y)
classes = list(le.classes_)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
proba = np.zeros((len(y), len(classes)), dtype=float)
for tr, te in skf.split(X, y_int):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=0)),
    ])
    pipe.fit(X[tr], y_int[tr])
    proba[te] = pipe.predict_proba(X[te])

print(f"dataset: n={len(y)}, classes={classes}")
print(f"target precision: {TARGET_PRECISION:.4f}\n")

for ci, c in enumerate(classes):
    p = proba[:, ci]
    is_c = (y_int == ci)

    # sweep thresholds = unique probability levels
    thresholds = np.unique(p)
    best = None
    for t in thresholds:
        pred = p >= t
        tp = int((pred & is_c).sum())
        fp = int((pred & ~is_c).sum())
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        if prec >= TARGET_PRECISION:
            recall = tp / int(is_c.sum())
            coverage = (tp + fp) / len(y)
            best = (t, prec, recall, coverage, tp, fp, int(is_c.sum()))
            break

    print(f"class {c}:")
    if best is None:
        print(f"  no threshold achieves precision >= {TARGET_PRECISION}")
    else:
        t, prec, recall, coverage, tp, fp, npos = best
        print(f"  threshold      : {t:.4f}")
        print(f"  precision      : {prec:.4f}  ({tp}/{tp + fp})")
        print(f"  recall of {c:<10}: {recall:.4f}  ({tp}/{npos})")
        print(f"  overall coverage: {coverage:.4f}  ({tp + fp}/{len(y)})")
    print()
