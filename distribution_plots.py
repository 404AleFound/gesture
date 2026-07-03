"""Velocity/curvature vs. displacement distribution plots.

For each of the 4 gestures x 4 directions = 16 combinations we plot:
    - background: every individual trajectory as a light line
    - foreground: mean curve computed by binning along the horizontal axis

Two versions per feature:
    - horizontal axis = x displacement
    - horizontal axis = y displacement

Output: 4 figures, each a 4x4 grid (rows=gestures, cols=directions).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from scrolls import DIRECTIONS, GESTURES, Scroll, load_scrolls
from features import curvatures, velocities

# Per-gesture color pair: (light, dark), same hue family.
GESTURE_COLOR_PAIRS: dict[str, tuple[str, str]] = {
    "LEFT_THUMB":  ("#a6cee3", "#1f6091"),
    "LEFT_INDEX":  ("#b2df8a", "#33a02c"),
    "RIGHT_THUMB": ("#fdbf6f", "#e37c1a"),
    "RIGHT_INDEX": ("#fb9a99", "#c0392b"),
}

# Displacement is measured relative to the first sample: p.x - x0 / p.y - y0.
# Bin edges are computed per cell from the actual data range, so each subplot
# can use the full axis without wasting canvas on the empty side.
N_BINS = 40


SeriesFn = Callable[[Scroll], tuple[np.ndarray, np.ndarray] | None]
# A SeriesFn returns (xs, ys) for one scroll where xs is the horizontal
# variable (already sliced to match ys) and ys is the feature series.


# ---------- feature series extractors ----------


def _velocity_series(scroll: Scroll, axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Instantaneous velocity vs. displacement (relative to first sample).

    Horizontal is (midpoint - first sample) on the chosen axis, so every
    trajectory starts at 0 on the horizontal axis and its sign shows the
    direction it moved.
    """
    path = scroll.path
    if len(path) < 2:
        return None
    vs = velocities(path)
    if not vs:
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


def _curvature_series(scroll: Scroll, axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Absolute curvature vs. displacement (relative to first sample)."""
    k = curvatures(scroll.path)
    if k.size == 0:
        return None
    x0, y0 = scroll.path[0].x, scroll.path[0].y
    horiz = np.asarray([
        (p.x - x0 if axis == "x" else p.y - y0) for p in scroll.path
    ])
    if horiz.size != k.size:
        return None
    return horiz, np.abs(k)


# ---------- binning ----------


def _bin_mean(
    xs: np.ndarray, ys: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Return per-bin mean of ys along `edges`; NaN for empty bins."""
    n = len(edges) - 1
    idx = np.digitize(xs, edges) - 1
    idx = np.clip(idx, 0, n - 1)
    out = np.full(n, np.nan)
    for b in range(n):
        mask = idx == b
        if mask.any():
            out[b] = ys[mask].mean()
    return out


def _draw_gesture_layer(
    ax,
    scrolls: list[Scroll],
    series_fn: SeriesFn,
    axis: str,
    color_light: str,
    color_dark: str,
    label: str,
    edges: np.ndarray,
) -> None:
    """Overlay one gesture's raw lines (light) + mean curve (dark) onto ax."""
    all_bin_means: list[np.ndarray] = []
    for scroll in scrolls:
        res = series_fn(scroll, axis)
        if res is None:
            continue
        xs, ys = res
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], color=color_light, alpha=0.28, linewidth=0.6)
        all_bin_means.append(_bin_mean(xs, ys, edges))

    if not all_bin_means:
        return

    stacked = np.vstack(all_bin_means)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(stacked, axis=0)
    valid = ~np.isnan(mean_curve)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers[valid], mean_curve[valid],
            color=color_dark, linewidth=2.0, label=label)


def _compute_cell_range(
    scrolls: list[Scroll], series_fn: SeriesFn, axis: str
) -> tuple[float, float, float, float] | None:
    """Scan a cell's scrolls to derive (x_min, x_max, y_min, y_max)."""
    x_lo = x_hi = None
    y_lo = y_hi = None
    for scroll in scrolls:
        res = series_fn(scroll, axis)
        if res is None:
            continue
        xs, ys = res
        if xs.size == 0:
            continue
        xmn, xmx = float(xs.min()), float(xs.max())
        ymn, ymx = float(ys.min()), float(ys.max())
        x_lo = xmn if x_lo is None else min(x_lo, xmn)
        x_hi = xmx if x_hi is None else max(x_hi, xmx)
        y_lo = ymn if y_lo is None else min(y_lo, ymn)
        y_hi = ymx if y_hi is None else max(y_hi, ymx)
    if x_lo is None:
        return None
    return x_lo, x_hi, y_lo, y_hi


def _pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
    """Add a small margin on both sides; ensure lo < hi even if lo==hi."""
    if lo == hi:
        return lo - 0.5, hi + 0.5
    m = (hi - lo) * frac
    return lo - m, hi + m


def _plot_grid(
    scrolls: list[Scroll],
    series_fn: SeriesFn,
    axis: str,
    ylabel: str,
    suptitle: str,
    out_path: Path,
) -> None:
    """2x2 grid keyed by direction; each cell overlays all 4 gestures.

    Every cell auto-scales its own x/y axes to fit only its own data.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for di, direction in enumerate(DIRECTIONS):
        ax = axes[di // 2, di % 2]
        cell_scrolls = [s for s in scrolls if s.direction == direction]

        # First pass: figure out the horizontal range for this cell,
        # then build a set of bin edges tied to that range.
        rng = _compute_cell_range(cell_scrolls, series_fn, axis)
        if rng is None:
            ax.set_title(f"{direction} (n=0)", fontsize=10)
            ax.axis("off")
            continue
        x_lo, x_hi, y_lo, y_hi = rng
        x_lo_p, x_hi_p = _pad(x_lo, x_hi)
        y_lo_p, y_hi_p = _pad(y_lo, y_hi)
        edges = np.linspace(x_lo_p, x_hi_p, N_BINS + 1)

        # Second pass: actually draw per-gesture layers using those edges.
        for gesture in GESTURES:
            light, dark = GESTURE_COLOR_PAIRS[gesture]
            subset = [s for s in cell_scrolls if s.tag == gesture]
            _draw_gesture_layer(
                ax, subset, series_fn, axis, light, dark, gesture, edges
            )

        ax.set_title(f"{direction} (n={len(cell_scrolls)})", fontsize=10)
        ax.set_xlim(x_lo_p, x_hi_p)
        ax.set_ylim(y_lo_p, y_hi_p)
        if x_lo_p <= 0 <= x_hi_p:
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


def plot_all(scrolls: list[Scroll], out_dir: str | Path = "figs/distribution") -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_grid(scrolls, _velocity_series, "x",
               ylabel="velocity (norm./ms)",
               suptitle="Velocity vs. Δx (relative to start)",
               out_path=out_dir / "velocity_vs_x.png")
    _plot_grid(scrolls, _velocity_series, "y",
               ylabel="velocity (norm./ms)",
               suptitle="Velocity vs. Δy (relative to start)",
               out_path=out_dir / "velocity_vs_y.png")
    _plot_grid(scrolls, _curvature_series, "x",
               ylabel="|curvature|",
               suptitle="Curvature vs. Δx (relative to start)",
               out_path=out_dir / "curvature_vs_x.png")
    _plot_grid(scrolls, _curvature_series, "y",
               ylabel="|curvature|",
               suptitle="Curvature vs. Δy (relative to start)",
               out_path=out_dir / "curvature_vs_y.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python distribution_plots.py <file.jsonl>", file=sys.stderr)
        sys.exit(1)
    scrolls = load_scrolls(sys.argv[1])
    print(f"loaded {len(scrolls)} scrolls")
    plot_all(scrolls)
    print("done: figs/distribution/")
