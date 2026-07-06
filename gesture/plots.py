"""All visualization code for the project.

Four top-level entry points:

    plot_scrolls(scrolls, out_dir)              -- trajectory views
    plot_features(scrolls, out_dir)             -- feature boxplots
    plot_distributions(scrolls, out_dir)        -- velocity/curvature vs displacement
    plot_classifier_results(results, out_dir)   -- confusion matrices + importances

`plot_all(scrolls, out_dir)` runs the first three in one go.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import numpy as np

from .scrolls import DIRECTIONS, GESTURES, IPHONE_ASPECT, Scroll
from .feature import Features, compute, curvatures, velocities

if TYPE_CHECKING:
    from eval import ClassifierResult


# ============================================================
# 1. Trajectory views (all / by_gesture / by_direction)
# ============================================================

# (panel-key or None, color-key). None means "all scrolls in one axis".
_SCROLL_VIEWS = {
    "all":          (None,                   lambda s: s.tag),
    "by_gesture":   (lambda s: s.tag,        lambda s: s.direction),
    "by_direction": (lambda s: s.direction,  lambda s: s.tag),
}
_SCROLL_PANELS = {"by_gesture": GESTURES, "by_direction": DIRECTIONS}


def plot_scrolls(
    scrolls: list[Scroll],
    out_dir: str | Path = "figs/scrolls",
    show: bool = False,
) -> None:
    """Render the 3 trajectory views (all / by_gesture / by_direction)."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    def setup(ax, title):
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect(IPHONE_ASPECT)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.4)

    def draw(ax, subset, color_of, palette):
        seen: set[str] = set()
        for s in subset:
            if not s.path:
                continue
            key = color_of(s)
            xs = [p.x for p in s.path]; ys = [p.y for p in s.path]
            label = key if key not in seen else None
            seen.add(key)
            c = cmap(palette.index(key) % 10)
            ax.plot(xs, ys, color=c, alpha=0.55, linewidth=1, label=label)
            ax.scatter([xs[0]], [ys[0]], color=c, s=15, marker="o")
            ax.scatter([xs[-1]], [ys[-1]], color=c, s=15, marker="x")
        if seen:
            ax.legend(loc="lower right", fontsize=8)

    for view, (panel_of, color_of) in _SCROLL_VIEWS.items():
        palette = list(GESTURES if color_of(scrolls[0]) in GESTURES else DIRECTIONS) \
            if scrolls else list(GESTURES)
        if panel_of is None:
            fig, ax = plt.subplots(figsize=(6, 6 * IPHONE_ASPECT / 2))
            draw(ax, scrolls, color_of, palette)
            setup(ax, f"All scrolls (n={len(scrolls)})")
            fig.tight_layout()
        else:
            panels = _SCROLL_PANELS[view]
            fig, axes = plt.subplots(2, 2, figsize=(7, 11))
            for ax, key in zip(axes.flat, panels):
                subset = [s for s in scrolls if panel_of(s) == key]
                draw(ax, subset, color_of, palette)
                setup(ax, f"{key} (n={len(subset)})")
            fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.05,
                                wspace=0.15, hspace=0.15)

        fig.savefig(out_dir / f"{view}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


# ============================================================
# 2. Feature boxplots (length / velocity / displacement / shape)
# ============================================================

# Directions colored consistently across every subplot.
_DIRECTION_COLORS: dict[str, str] = {
    "RIGHT": "#4c72b0",  # blue
    "LEFT":  "#dd8452",  # orange
    "DOWN":  "#55a868",  # green
    "UP":    "#c44e52",  # red
}

FeatureGetter = Callable[[Features], float]


def _aggregate_features(
    scrolls: list[Scroll],
) -> dict[tuple[str, str], list[Features]]:
    """Group features by (direction, gesture)."""
    agg: dict[tuple[str, str], list[Features]] = {
        (d, g): [] for d in DIRECTIONS for g in GESTURES
    }
    for scroll in scrolls:
        agg[(scroll.direction, scroll.tag)].append(compute(scroll))
    return agg


def _grouped_boxplot(ax, agg, getter, title, ylabel, zero_line=False) -> None:
    """Grouped boxplot: x=gestures, per group=directions."""
    n_d = len(DIRECTIONS)
    box_width = 0.8 / n_d
    for di, direction in enumerate(DIRECTIONS):
        positions, data = [], []
        for gi, gesture in enumerate(GESTURES):
            positions.append(gi + (di - (n_d - 1) / 2) * box_width)
            data.append([getter(f) for f in agg[(direction, gesture)]])
        color = _DIRECTION_COLORS[direction]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.85,
            patch_artist=True,
            flierprops=dict(marker="o", markersize=2, markerfacecolor=color,
                            markeredgecolor=color, alpha=0.5),
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.75),
        )
        bp["boxes"][0].set_label(direction)

    if zero_line:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks(range(len(GESTURES)))
    ax.set_xticklabels(GESTURES, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)


def _feature_legend(fig) -> None:
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=_DIRECTION_COLORS[d], edgecolor=_DIRECTION_COLORS[d], label=d)
        for d in DIRECTIONS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))


# Panel specs: (getter, title, ylabel, zero_line?). None entries -> blank axis.
_FEATURE_PANELS: dict[str, tuple[tuple, ...]] = {
    "length": (
        (lambda f: f.length, "Length", "length (norm.)", False),
    ),
    "velocity": (
        (lambda f: f.velocity_max, "Velocity max", "v_max (norm./ms)", False),
        (lambda f: f.velocity_mean, "Velocity mean", "v_mean (norm./ms)", False),
        (lambda f: f.velocity_std, "Velocity std", "v_std (norm./ms)", False),
    ),
    "displacement": (
        (lambda f: f.disp_total_dx, "Total dx (signed)", "dx (norm.)", True),
        (lambda f: f.disp_max_dx,   "Max dx (span)",     "dx (norm.)", False),
        (lambda f: f.disp_total_dy, "Total dy (signed)", "dy (norm.)", True),
        (lambda f: f.disp_max_dy,   "Max dy (span)",     "dy (norm.)", False),
    ),
    "shape": (
        (lambda f: f.curvature_rmse, "RMSE (fit residual)", "rmse (norm.)", False),
        (lambda f: f.curvature_max,  "Curvature max",       "κ_max",        False),
        (lambda f: f.curvature_mean, "Curvature mean",      "κ_mean",       False),
    ),
}


def plot_features(
    scrolls: list[Scroll],
    out_dir: str | Path = "figs/features",
) -> None:
    """Boxplots for length/velocity/displacement/shape features."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _aggregate_features(scrolls)

    for name, panels in _FEATURE_PANELS.items():
        if len(panels) == 1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            _grouped_boxplot(ax, agg, *panels[0])
            _feature_legend(fig)
            fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.1)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for ax, spec in zip(axes.flat, panels):
                _grouped_boxplot(ax, agg, *spec)
            for ax in axes.flat[len(panels):]:
                ax.axis("off")
            _feature_legend(fig)
            fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07,
                                wspace=0.22, hspace=0.28)

        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 3. Velocity/curvature distributions vs displacement
# ============================================================

# Per-gesture color pair: (light background, dark foreground).
_GESTURE_COLOR_PAIRS: dict[str, tuple[str, str]] = {
    "LEFT_THUMB":  ("#a6cee3", "#1f6091"),
    "LEFT_INDEX":  ("#b2df8a", "#33a02c"),
    "RIGHT_THUMB": ("#fdbf6f", "#e37c1a"),
    "RIGHT_INDEX": ("#fb9a99", "#c0392b"),
}

_N_BINS = 40

SeriesFn = Callable[[Scroll, str], tuple[np.ndarray, np.ndarray] | None]


def _velocity_series(scroll: Scroll, axis: str):
    """Instantaneous velocity vs. displacement (relative to first sample)."""
    path = scroll.path
    if len(path) < 2:
        return None
    if not velocities(path):
        return None
    x0, y0 = path[0].x, path[0].y
    xs, ys, out_v = [], [], []
    for a, b in zip(path, path[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        xs.append(0.5 * (a.x + b.x) - x0)
        ys.append(0.5 * (a.y + b.y) - y0)
        out_v.append(((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5 / dt)
    if not out_v:
        return None
    horiz = np.asarray(xs if axis == "x" else ys)
    return horiz, np.asarray(out_v)


def _curvature_series(scroll: Scroll, axis: str):
    """Absolute curvature vs. displacement (relative to first sample)."""
    k = curvatures(scroll.path)
    if k.size == 0:
        return None
    x0, y0 = scroll.path[0].x, scroll.path[0].y
    horiz = np.asarray([(p.x - x0 if axis == "x" else p.y - y0) for p in scroll.path])
    if horiz.size != k.size:
        return None
    return horiz, np.abs(k)


def _bin_mean(xs: np.ndarray, ys: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-bin mean of ys along `edges`; NaN for empty bins."""
    n = len(edges) - 1
    idx = np.clip(np.digitize(xs, edges) - 1, 0, n - 1)
    out = np.full(n, np.nan)
    for b in range(n):
        mask = idx == b
        if mask.any():
            out[b] = ys[mask].mean()
    return out


def _pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
    if lo == hi:
        return lo - 0.5, hi + 0.5
    m = (hi - lo) * frac
    return lo - m, hi + m


def _cell_range(scrolls, series_fn, axis):
    x_lo = x_hi = y_lo = y_hi = None
    for s in scrolls:
        res = series_fn(s, axis)
        if res is None or res[0].size == 0:
            continue
        xs, ys = res
        xmn, xmx = float(xs.min()), float(xs.max())
        ymn, ymx = float(ys.min()), float(ys.max())
        x_lo = xmn if x_lo is None else min(x_lo, xmn)
        x_hi = xmx if x_hi is None else max(x_hi, xmx)
        y_lo = ymn if y_lo is None else min(y_lo, ymn)
        y_hi = ymx if y_hi is None else max(y_hi, ymx)
    if x_lo is None:
        return None
    return x_lo, x_hi, y_lo, y_hi


def _draw_gesture_layer(ax, scrolls, series_fn, axis, color_light, color_dark,
                        label, edges) -> None:
    all_means: list[np.ndarray] = []
    for s in scrolls:
        res = series_fn(s, axis)
        if res is None:
            continue
        xs, ys = res
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], color=color_light, alpha=0.28, linewidth=0.6)
        all_means.append(_bin_mean(xs, ys, edges))
    if not all_means:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(np.vstack(all_means), axis=0)
    valid = ~np.isnan(mean_curve)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers[valid], mean_curve[valid],
            color=color_dark, linewidth=2.0, label=label)


def _plot_distribution_grid(scrolls, series_fn, axis, ylabel, suptitle, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for di, direction in enumerate(DIRECTIONS):
        ax = axes[di // 2, di % 2]
        cell = [s for s in scrolls if s.direction == direction]
        rng = _cell_range(cell, series_fn, axis)
        if rng is None:
            ax.set_title(f"{direction} (n=0)", fontsize=10)
            ax.axis("off")
            continue
        x_lo, x_hi, y_lo, y_hi = rng
        x_lo, x_hi = _pad(x_lo, x_hi)
        y_lo, y_hi = _pad(y_lo, y_hi)
        edges = np.linspace(x_lo, x_hi, _N_BINS + 1)
        for gesture in GESTURES:
            light, dark = _GESTURE_COLOR_PAIRS[gesture]
            subset = [s for s in cell if s.tag == gesture]
            _draw_gesture_layer(ax, subset, series_fn, axis, light, dark, gesture, edges)

        ax.set_title(f"{direction} (n={len(cell)})", fontsize=10)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        if x_lo <= 0 <= x_hi:
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.tick_params(axis="both", labelsize=8)
        if di // 2 == 1:
            ax.set_xlabel(f"Δ{axis} (relative to start)", fontsize=9)
        if di % 2 == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.suptitle(suptitle, fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08,
                        wspace=0.18, hspace=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# (series_fn, axis, ylabel, suptitle-template) -> output filename
_DISTRIBUTION_SPECS = [
    (_velocity_series,  "x", "velocity (norm./ms)", "Velocity vs. Δx (relative to start)", "velocity_vs_x.png"),
    (_velocity_series,  "y", "velocity (norm./ms)", "Velocity vs. Δy (relative to start)", "velocity_vs_y.png"),
    (_curvature_series, "x", "|curvature|",         "Curvature vs. Δx (relative to start)", "curvature_vs_x.png"),
    (_curvature_series, "y", "|curvature|",         "Curvature vs. Δy (relative to start)", "curvature_vs_y.png"),
]


def plot_distributions(
    scrolls: list[Scroll],
    out_dir: str | Path = "figs/distribution",
) -> None:
    """4-figure velocity/curvature distributions vs Δx/Δy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn, axis, ylabel, suptitle, fname in _DISTRIBUTION_SPECS:
        _plot_distribution_grid(scrolls, fn, axis, ylabel, suptitle, out_dir / fname)


# ============================================================
# 4. Classifier evaluation figures
# ============================================================


def plot_classifier_results(
    results: dict[str, "ClassifierResult"],
    out_dir: str | Path = "figs/classifiers",
) -> None:
    """Confusion matrices + feature importances for a set of trained classifiers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrices(results, out_dir / "confusion_matrices.png")
    _plot_feature_importances(results, out_dir / "feature_importance.png")


def _plot_confusion_matrices(results, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    items = list(results.items())
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))

    im = None
    for ax, (name, r) in zip(axes.flat, items):
        cm = r.confusion_matrix
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{cm[i, j]}\n{cm_norm[i, j]*100:.0f}%"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

        ax.set_xticks(range(len(r.labels)))
        ax.set_yticks(range(len(r.labels)))
        ax.set_xticklabels(r.labels, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(r.labels, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"{name}\nacc = {r.acc_mean:.3f}", fontsize=9)

    for ax in axes.flat[len(items):]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                     label="row-normalized frequency")
    fig.suptitle("Confusion matrices (5-fold OOF)", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importances(results, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    have = [r for r in results.values() if r.feature_importances is not None]
    if not have:
        return

    fig, axes = plt.subplots(len(have), 1, figsize=(10, 3.2 * len(have)), squeeze=False)
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


# ============================================================
# Convenience: run every scroll-based figure
# ============================================================


def plot_all(scrolls: list[Scroll], out_root: str | Path = "figs") -> None:
    root = Path(out_root)
    plot_scrolls(scrolls, out_dir=root / "scrolls")
    plot_features(scrolls, out_dir=root / "features")
    plot_distributions(scrolls, out_dir=root / "distribution")


