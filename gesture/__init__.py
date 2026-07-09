"""Gesture recognition library.

Everything under `gesture/` is passive: data classes, feature
computations, model wrappers, plotting library functions. Action scripts
(train.py, eval.py, plot.py) live at the project root and orchestrate
what's defined here.

Public API:

    from gesture import (
        # data model
        Scroll, PathPoint, load_scrolls, GESTURES, DIRECTIONS,
        # features
        Features, compute,
        # dataset
        GestureDataset, FEATURE_NAMES,
        # models
        Classifier, DecisionTree, RandomForest, NaiveBayes, MLP, KNN,
        ALL_MODELS,
    )

Sub-modules that expose additional helpers (e.g. `gesture.plots` for the
figure functions or `gesture.feature` for the low-level trajectory math)
can still be imported directly.
"""

from .scrolls import (
    DIRECTIONS,
    GESTURES,
    IPHONE_ASPECT,
    PathPoint,
    Scroll,
    load_scrolls,
)
from .feature import Features, compute
from .dataset import FEATURE_NAMES, HAND_LABELS, GestureDataset
from .model import (
    ALL_MODELS,
    Classifier,
    DecisionTree,
    KNN,
    MLP,
    NaiveBayes,
    RandomForest,
)
from .logging import (
    current_run_dir,
    get_logger,
    install_run_logger,
)

__all__ = [
    # scrolls
    "DIRECTIONS",
    "GESTURES",
    "IPHONE_ASPECT",
    "PathPoint",
    "Scroll",
    "load_scrolls",
    # features
    "Features",
    "compute",
    # dataset
    "FEATURE_NAMES",
    "HAND_LABELS",
    "GestureDataset",
    # models
    "ALL_MODELS",
    "Classifier",
    "DecisionTree",
    "KNN",
    "MLP",
    "NaiveBayes",
    "RandomForest",
    # logging
    "install_run_logger",
    "current_run_dir",
    "get_logger",
]
