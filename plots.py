"""Task 2 feature-distribution plots.

Layout for every continuous feature:
    x axis  : 4 gestures (LEFT_THUMB / RIGHT_THUMB / LEFT_INDEX / RIGHT_INDEX)
    boxes   : 4 directions side-by-side in each gesture group
    y axis  : feature value

Grouping by gesture makes intra-gesture consistency across directions easy
to eyeball: if the 4 direction boxes cluster tightly for a gesture, that's
a candidate discriminating feature.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable

import numpy as np

from scrolls import DIRECTIONS, GESTURES, Scroll
from features import TrajectoryFeatures, compute

# Directions colored consistently across every subplot.
DIRECTION_COLORS: dict[str, str] = {
    "RIGHT": "#4c72b0",  # blue
    "LEFT": "#dd8452",   # orange
    "DOWN": "#55a868",   # green
    "UP": "#c44e52",     # red
}

FeatureGetter = Callable[[TrajectoryFeatures], float]


def aggregate(
    scrolls: list[Scroll],
) -> dict[tuple[str, str], list[TrajectoryFeatures]]:
    """Group features by (direction, gesture)."""
    agg: dict[tuple[str, str], list[TrajectoryFeatures]] = {
        (d, g): [] for d in DIRECTIONS for g in GESTURES
    }
    for scroll in scrolls:
        feat = compute(scroll)
        agg[(scroll.direction, scroll.tag)].append(feat)
    return agg


def _values(
    agg: dict[tuple[str, str], list[TrajectoryFeatures]],
    getter: FeatureGetter,
    direction: str,
    gesture: str,
) -> list[float]:
    return [getter(f) for f in agg[(direction, gesture)]]


def _grouped_boxplot(
    ax,
    agg: dict[tuple[str, str], list[TrajectoryFeatures]],
    getter: FeatureGetter,
    title: str,
    ylabel: str,
    zero_line: bool = False,
) -> None:
    """Draw a grouped boxplot: x=gestures, per group=directions."""
    n_d = len(DIRECTIONS)
    group_width = 0.8
    box_width = group_width / n_d
    xticks: list[int] = list(range(len(GESTURES)))

    for di, direction in enumerate(DIRECTIONS):
        positions = []
        data = []
        for gi, gesture in enumerate(GESTURES):
            pos = gi + (di - (n_d - 1) / 2) * box_width
            positions.append(pos)
            data.append(_values(agg, getter, direction, gesture))
        color = DIRECTION_COLORS[direction]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=True,
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

    ax.set_xticks(xticks)
    ax.set_xticklabels(GESTURES, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)


def _add_legend(fig, ncol: int = 4) -> None:
    """Single figure-level legend for the 4 directions."""
    handles = [
        __import__("matplotlib").patches.Patch(
            facecolor=DIRECTION_COLORS[d], edgecolor=DIRECTION_COLORS[d], label=d
        )
        for d in DIRECTIONS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )


# ---------- 4 feature figures ----------


def plot_length(agg, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_boxplot(ax, agg, lambda f: f.length, "Length", "length (norm.)")
    _add_legend(fig)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity(agg, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        (axes[0, 0], lambda f: f.velocity_max, "Velocity max", "v_max (norm./ms)"),
        (axes[0, 1], lambda f: f.velocity_mean, "Velocity mean", "v_mean (norm./ms)"),
        (axes[1, 0], lambda f: f.velocity_std, "Velocity std", "v_std (norm./ms)"),
    ]
    for ax, getter, title, ylabel in specs:
        _grouped_boxplot(ax, agg, getter, title, ylabel)
    axes[1, 1].axis("off")
    _add_legend(fig)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07, wspace=0.22, hspace=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_displacement(agg, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _grouped_boxplot(axes[0, 0], agg, lambda f: f.total_dx,
                     "Total dx (signed)", "dx (norm.)", zero_line=True)
    _grouped_boxplot(axes[0, 1], agg, lambda f: f.max_dx,
                     "Max dx (span)", "dx (norm.)")
    _grouped_boxplot(axes[1, 0], agg, lambda f: f.total_dy,
                     "Total dy (signed)", "dy (norm.)", zero_line=True)
    _grouped_boxplot(axes[1, 1], agg, lambda f: f.max_dy,
                     "Max dy (span)", "dy (norm.)")
    _add_legend(fig)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07, wspace=0.22, hspace=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shape(agg, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _grouped_boxplot(axes[0, 0], agg, lambda f: f.rmse,
                     "RMSE (fit residual)", "rmse (norm.)")
    _grouped_boxplot(axes[0, 1], agg, lambda f: f.curvature_max,
                     "Curvature max", "κ_max")
    _grouped_boxplot(axes[1, 0], agg, lambda f: f.curvature_mean,
                     "Curvature mean", "κ_mean")
    axes[1, 1].axis("off")
    _add_legend(fig)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07, wspace=0.22, hspace=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(scrolls: list[Scroll], out_dir: str | Path = "figs/features") -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(scrolls)
    plot_length(agg, out_dir / "length.png")
    plot_velocity(agg, out_dir / "velocity.png")
    plot_displacement(agg, out_dir / "displacement.png")
    plot_shape(agg, out_dir / "shape.png")


if __name__ == "__main__":
    from scrolls import load_scrolls
    if len(sys.argv) < 2:
        print("usage: python feature_plots.py <file.jsonl>", file=sys.stderr)
        sys.exit(1)
    scrolls = load_scrolls(sys.argv[1])
    print(f"loaded {len(scrolls)} scrolls")
    plot_all(scrolls)
    print("done: figs/features/")
