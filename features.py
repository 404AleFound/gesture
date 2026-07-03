import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from scrolls import Scroll, PathPoint

POLY_DEGREE = 4


@dataclass
class TrajectoryFeatures:
    length: float
    displacement: float
    direction_rad: float
    duration_ms: int
    point_count: int
    velocity_max: float
    velocity_mean: float
    velocity_std: float
    total_dx: float
    total_dy: float
    max_dx: float
    max_dy: float
    rmse: float
    curvature_max: float
    curvature_mean: float
    convex_orientation: int

    @property
    def direction_deg(self) -> float:
        return math.degrees(self.direction_rad)

    @property
    def straightness(self) -> float:
        if self.length == 0:
            return 0.0
        return self.displacement / self.length


def length(path: list[PathPoint]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b.x - a.x, b.y - a.y)
    return total


def displacement(path: list[PathPoint]) -> float:
    if len(path) < 2:
        return 0.0
    a, b = path[0], path[-1]
    return math.hypot(b.x - a.x, b.y - a.y)


def direction(path: list[PathPoint]) -> float:
    if len(path) < 2:
        return 0.0
    a, b = path[0], path[-1]
    return math.atan2(b.y - a.y, b.x - a.x)


def total_displacement_xy(path: list[PathPoint]) -> tuple[float, float]:
    if len(path) < 2:
        return 0.0, 0.0
    a, b = path[0], path[-1]
    return b.x - a.x, b.y - a.y


def max_displacement_xy(path: list[PathPoint]) -> tuple[float, float]:
    if not path:
        return 0.0, 0.0
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    return max(xs) - min(xs), max(ys) - min(ys)


def rotate_to_principal_axis(path: list[PathPoint]) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates so first->last vector aligns with +x axis.

    Returns (u, v) where u is the coordinate along the principal axis and v
    is perpendicular. This makes f(u) well-defined for polynomial fitting.
    """
    if len(path) < 2:
        return np.array([p.x for p in path]), np.array([p.y for p in path])
    xs = np.array([p.x for p in path])
    ys = np.array([p.y for p in path])
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    theta = math.atan2(dy, dx)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xs0 = xs - xs[0]
    ys0 = ys - ys[0]
    u = cos_t * xs0 + sin_t * ys0
    v = -sin_t * xs0 + cos_t * ys0
    return u, v


MIN_FIT_SPAN = 0.05


def fit_polynomial(path: list[PathPoint], degree: int = POLY_DEGREE):
    """Fit v = p(u) in the rotated frame.

    Returns (u, v, poly) where poly is a numpy.polynomial.Polynomial object,
    or None if the path is too short to fit reliably.
    """
    if len(path) < degree + 2:
        return None
    u, v = rotate_to_principal_axis(path)
    if np.ptp(u) < MIN_FIT_SPAN:
        return None
    poly = np.polynomial.Polynomial.fit(u, v, degree)
    return u, v, poly


def rmse(path: list[PathPoint]) -> float:
    fit = fit_polynomial(path)
    if fit is None:
        return 0.0
    u, v, poly = fit
    residuals = v - poly(u)
    return float(np.sqrt(np.mean(residuals ** 2)))


def curvatures(path: list[PathPoint]) -> np.ndarray:
    fit = fit_polynomial(path)
    if fit is None:
        return np.array([])
    u, _v, poly = fit
    d1 = poly.deriv(1)
    d2 = poly.deriv(2)
    fp = d1(u)
    fpp = d2(u)
    return fpp / np.power(1.0 + fp ** 2, 1.5)


def convex_orientation(path: list[PathPoint], eps: float = 1e-6) -> int:
    """Signed convex side of the curve, normalized by primary motion direction.

    +1 / -1 correspond to opposite convex sides; 0 when the curve is too
    close to a straight line to decide.
    """
    if len(path) < 3:
        return 0
    p_start = path[0]
    p_end = path[-1]
    p_mid = path[len(path) // 2]
    v1x, v1y = p_start.x - p_mid.x, p_start.y - p_mid.y
    v2x, v2y = p_end.x - p_mid.x, p_end.y - p_mid.y
    z = v2x * v1y - v2y * v1x
    if abs(z) < eps:
        return 0
    dx = p_end.x - p_start.x
    dy = p_end.y - p_start.y
    norm = dy if abs(dy) >= abs(dx) else dx
    if norm == 0:
        return 0
    return int(math.copysign(1, norm * z))


def curvature_stats(path: list[PathPoint]) -> tuple[float, float]:
    k = curvatures(path)
    if k.size == 0:
        return 0.0, 0.0
    abs_k = np.abs(k)
    return float(abs_k.max()), float(abs_k.mean())


def velocities(path: list[PathPoint]) -> list[float]:
    result: list[float] = []
    for a, b in zip(path, path[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        result.append(math.hypot(b.x - a.x, b.y - a.y) / dt)
    return result


def velocity_stats(path: list[PathPoint]) -> tuple[float, float, float]:
    """Return (v_max, v_mean, v_std).

    v_mean is the time-weighted average = total_length / total_duration,
    matching the spec in tasks.md and avoiding sampling-rate bias. When
    the total duration is zero (bad data), fall back to the arithmetic
    mean of the per-segment velocities.
    """
    vs = velocities(path)
    if not vs:
        return 0.0, 0.0, 0.0
    v_max = max(vs)
    total_duration = path[-1].t - path[0].t
    if total_duration > 0:
        v_mean = length(path) / total_duration
    else:
        v_mean = sum(vs) / len(vs)
    v_std = math.sqrt(sum((v - v_mean) ** 2 for v in vs) / len(vs))
    return v_max, v_mean, v_std


def plot_length_histograms(
    scrolls: list[Scroll],
    out_dir: str | Path = "figs/length",
    n_bins: int = 20,
    show: bool = False,
):
    """Plot 5 length histograms: one per tag plus a combined comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths_by_tag: dict[str, list[float]] = {}
    for scroll in scrolls:
        lengths_by_tag.setdefault(scroll.tag, []).append(length(scroll.path))
    if not lengths_by_tag:
        return

    tags = sorted(lengths_by_tag)
    all_lengths = [v for vs in lengths_by_tag.values() for v in vs]
    lo, hi = min(all_lengths), max(all_lengths)
    bins = np.linspace(lo, hi, n_bins + 1)
    cmap = plt.get_cmap("tab10")
    color_by_tag = {tag: cmap(i % 10) for i, tag in enumerate(tags)}

    for tag in tags:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(lengths_by_tag[tag], bins=bins, color=color_by_tag[tag], edgecolor="black")
        ax.set_title(f"Length distribution — {tag} (n={len(lengths_by_tag[tag])})")
        ax.set_xlabel("length (normalized)")
        ax.set_ylabel("count")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"length_{tag}.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tag in tags:
        ax.hist(
            lengths_by_tag[tag],
            bins=bins,
            density=True,
            histtype="step",
            label=f"{tag} (n={len(lengths_by_tag[tag])})",
            color=color_by_tag[tag],
            linewidth=2,
        )
    ax.set_title("Length distribution — all tags (density)")
    ax.set_xlabel("length (normalized)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "length_all.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    means = [float(np.mean(lengths_by_tag[tag])) for tag in tags]
    stds = [float(np.std(lengths_by_tag[tag])) for tag in tags]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        tags,
        means,
        yerr=stds,
        capsize=6,
        color=[color_by_tag[t] for t in tags],
        edgecolor="black",
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Mean length by tag (error bar = std)")
    ax.set_ylabel("length (normalized)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "length_mean.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


MEAN_SKIP_FIELDS = {"direction_rad"}


def mean_features_by_tag(scrolls: list[Scroll]) -> dict[str, dict[str, float]]:
    """Compute per-tag mean of every numeric feature.

    Returns {tag: {feature_name: mean_value, ..., "count": n}}.
    Fields in MEAN_SKIP_FIELDS (e.g. angular) are omitted.
    """
    features_by_tag: dict[str, list[TrajectoryFeatures]] = {}
    for scroll in scrolls:
        features_by_tag.setdefault(scroll.tag, []).append(compute(scroll))

    field_names = [
        f.name for f in fields(TrajectoryFeatures) if f.name not in MEAN_SKIP_FIELDS
    ]
    result: dict[str, dict[str, float]] = {}
    for tag, feats in features_by_tag.items():
        means: dict[str, float] = {"count": float(len(feats))}
        for name in field_names:
            values = [getattr(f, name) for f in feats]
            means[name] = float(sum(values) / len(values))
        result[tag] = means
    return result


def print_mean_features_by_tag(scrolls: list[Scroll]) -> None:
    means = mean_features_by_tag(scrolls)
    if not means:
        print("no scrolls")
        return
    tags = sorted(means)
    field_names = ["count"] + [
        f.name for f in fields(TrajectoryFeatures) if f.name not in MEAN_SKIP_FIELDS
    ]
    col_width = max(len(n) for n in field_names) + 2
    tag_width = max(len(t) for t in tags) + 2
    header = "feature".ljust(col_width) + "".join(t.ljust(tag_width) for t in tags)
    print(header)
    print("-" * len(header))
    for name in field_names:
        row = name.ljust(col_width)
        for t in tags:
            row += f"{means[t][name]:.4g}".ljust(tag_width)
        print(row)


def compute(scroll: Scroll) -> TrajectoryFeatures:
    v_max, v_mean, v_std = velocity_stats(scroll.path)
    total_dx, total_dy = total_displacement_xy(scroll.path)
    max_dx, max_dy = max_displacement_xy(scroll.path)
    k_max, k_mean = curvature_stats(scroll.path)
    return TrajectoryFeatures(
        length=length(scroll.path),
        displacement=displacement(scroll.path),
        direction_rad=direction(scroll.path),
        duration_ms=scroll.duration_ms,
        point_count=scroll.count,
        velocity_max=v_max,
        velocity_mean=v_mean,
        velocity_std=v_std,
        total_dx=total_dx,
        total_dy=total_dy,
        max_dx=max_dx,
        max_dy=max_dy,
        rmse=rmse(scroll.path),
        curvature_max=k_max,
        curvature_mean=k_mean,
        convex_orientation=convex_orientation(scroll.path),
    )
