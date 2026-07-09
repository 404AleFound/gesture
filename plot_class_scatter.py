"""Check whether merging LEFT_INDEX + RIGHT_INDEX into MID_HAND is defensible.

Rationale: the `hand1` label mode assumes food-finger gestures are "neutral"
(fingers roam over the whole screen with no held-side bias), so their
starting positions should overlap. If LEFT_INDEX and RIGHT_INDEX turn out
to be well-separated on position_begin_x, merging them would erase a real
signal and hurt accuracy. This script visualizes and quantifies that
overlap so the modeling choice rests on data instead of intuition.

Outputs (default figs/hand1_check/):
- begin_scatter.png         -- 2D scatter of (position_begin_x, position_begin_y), one dot per scroll, colored by tag
- begin_x_hist.png          -- 4 rows x 4 cols: position_begin_x histogram per (tag, direction)
- begin_x_by_direction.png  -- overlaid tag histograms per direction, easy to eyeball LEFT_INDEX vs RIGHT_INDEX
- report.txt                -- printed + saved KS statistic and simple bin-overlap between LEFT_INDEX and RIGHT_INDEX

Usage:
    python plot_class_scatter.py data/gestures-all.jsonl [--out-dir figs/hand1_check]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gesture.plots import _GESTURE_COLOR_PAIRS
from gesture.scrolls import DIRECTIONS, GESTURES, IPHONE_ASPECT, load_scrolls


# ---------- data prep ----------


def _positions(scrolls) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {tag: (xs, ys)} of start positions."""
    xs: dict[str, list[float]] = {g: [] for g in GESTURES}
    ys: dict[str, list[float]] = {g: [] for g in GESTURES}
    for s in scrolls:
        if not s.path:
            continue
        xs[s.tag].append(s.path[0].x)
        ys[s.tag].append(s.path[0].y)
    return {g: (np.asarray(xs[g]), np.asarray(ys[g])) for g in GESTURES}


def _positions_by_dir(scrolls) -> dict[tuple[str, str], np.ndarray]:
    """{(tag, direction): begin_x array}. Directions splittable per panel."""
    out: dict[tuple[str, str], list[float]] = {
        (g, d): [] for g in GESTURES for d in DIRECTIONS
    }
    for s in scrolls:
        if s.path:
            out[(s.tag, s.direction)].append(s.path[0].x)
    return {k: np.asarray(v) for k, v in out.items()}


# ---------- overlap metrics ----------


def _bin_overlap(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """L1 histogram overlap in [0, 1]: 1 = identical, 0 = disjoint.

    Both arrays are histogrammed over their combined range, normalized to
    sum to 1, and the per-bin min is summed. This is a very rough analogue
    of the Bhattacharyya coefficient that doesn't need a distributional
    assumption.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    return float(np.minimum(pa, pb).sum())


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic without scipy.

    Big number = distributions differ, small = they overlap. Not returning
    a p-value here; the statistic alone is enough for a sanity check.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


# ---------- plots ----------


def _scatter_begin(pos, out_path: Path) -> None:
    """2D scatter of start positions; highlight LEFT_INDEX and RIGHT_INDEX."""
    fig, ax = plt.subplots(figsize=(6, 6 * IPHONE_ASPECT / 2))
    for tag in GESTURES:
        xs, ys = pos[tag]
        light, dark = _GESTURE_COLOR_PAIRS[tag]
        # Thumbs get dimmer so INDEX classes pop.
        alpha = 0.75 if "INDEX" in tag else 0.30
        size = 10 if "INDEX" in tag else 6
        ax.scatter(xs, ys, s=size, color=dark, alpha=alpha, label=f"{tag} (n={xs.size})")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect(IPHONE_ASPECT)
    ax.set_xlabel("position_begin_x")
    ax.set_ylabel("position_begin_y")
    ax.set_title("Start-position scatter\n(if LEFT_INDEX and RIGHT_INDEX overlap, MID_HAND is justified)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hist_grid(pos_by_dir, out_path: Path) -> None:
    """4x4 grid: rows = tag, cols = direction, cell = begin_x histogram."""
    fig, axes = plt.subplots(len(GESTURES), len(DIRECTIONS),
                             figsize=(2.6 * len(DIRECTIONS), 1.7 * len(GESTURES)),
                             sharex=True, sharey=False)
    edges = np.linspace(0, 1, 25)
    for r, tag in enumerate(GESTURES):
        _, dark = _GESTURE_COLOR_PAIRS[tag]
        for c, direction in enumerate(DIRECTIONS):
            ax = axes[r, c]
            data = pos_by_dir[(tag, direction)]
            ax.hist(data, bins=edges, color=dark, alpha=0.85)
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            if r == 0:
                ax.set_title(direction, fontsize=9)
            if c == 0:
                ax.set_ylabel(tag, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.set_xlim(0, 1)
    fig.suptitle("position_begin_x by (tag, direction)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_by_direction(pos_by_dir, out_path: Path) -> None:
    """One panel per direction, all 4 tag histograms overlaid."""
    fig, axes = plt.subplots(1, len(DIRECTIONS),
                             figsize=(3.6 * len(DIRECTIONS), 3),
                             sharex=True, sharey=True)
    edges = np.linspace(0, 1, 25)
    for c, direction in enumerate(DIRECTIONS):
        ax = axes[c]
        for tag in GESTURES:
            _, dark = _GESTURE_COLOR_PAIRS[tag]
            data = pos_by_dir[(tag, direction)]
            ax.hist(data, bins=edges, color=dark, alpha=0.45,
                    histtype="stepfilled", label=tag)
            ax.hist(data, bins=edges, color=dark,
                    histtype="step", linewidth=1.2)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(direction, fontsize=10)
        ax.set_xlabel("position_begin_x", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if c == 0:
            ax.set_ylabel("count")
        ax.set_xlim(0, 1)
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("position_begin_x per direction (overlay all tags)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- driver ----------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", help="gestures jsonl to analyze")
    p.add_argument("--out-dir", default="figs/hand1_check")
    args = p.parse_args()

    scrolls = list(load_scrolls(args.jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = _positions(scrolls)
    pos_by_dir = _positions_by_dir(scrolls)

    _scatter_begin(pos, out_dir / "begin_scatter.png")
    _hist_grid(pos_by_dir, out_dir / "begin_x_hist.png")
    _overlay_by_direction(pos_by_dir, out_dir / "begin_x_by_direction.png")

    li_x, _ = pos["LEFT_INDEX"]
    ri_x, _ = pos["RIGHT_INDEX"]
    lt_x, _ = pos["LEFT_THUMB"]
    rt_x, _ = pos["RIGHT_THUMB"]

    lines: list[str] = []
    def emit(*msg): line = " ".join(str(m) for m in msg); print(line); lines.append(line)

    emit(f"scrolls: {len(scrolls)}")
    emit(f"per-tag n: " + ", ".join(f"{g}={pos[g][0].size}" for g in GESTURES))
    emit("")
    emit("=== position_begin_x: LEFT_INDEX vs RIGHT_INDEX ===")
    emit(f"  LEFT_INDEX  mean={li_x.mean():.3f}  std={li_x.std():.3f}  n={li_x.size}")
    emit(f"  RIGHT_INDEX mean={ri_x.mean():.3f}  std={ri_x.std():.3f}  n={ri_x.size}")
    emit(f"  KS statistic          = {_ks_stat(li_x, ri_x):.3f}  (0 = identical, 1 = disjoint)")
    emit(f"  bin overlap (25 bins) = {_bin_overlap(li_x, ri_x, bins=25):.3f}  (1 = identical, 0 = disjoint)")
    emit("")
    emit("=== reference: THUMB pairs (should be well-separated) ===")
    emit(f"  LT vs RT  KS={_ks_stat(lt_x, rt_x):.3f}  overlap={_bin_overlap(lt_x, rt_x, bins=25):.3f}")
    emit(f"  LT vs LI  KS={_ks_stat(lt_x, li_x):.3f}  overlap={_bin_overlap(lt_x, li_x, bins=25):.3f}")
    emit(f"  RT vs RI  KS={_ks_stat(rt_x, ri_x):.3f}  overlap={_bin_overlap(rt_x, ri_x, bins=25):.3f}")
    emit("")
    emit("interpretation:")
    emit("  - LI vs RI overlap high (>= ~0.6) & KS low (< ~0.25) -> merging into MID_HAND is defensible.")
    emit("  - overlap low (< ~0.3) & KS high (> ~0.4)            -> the two index classes are separable;")
    emit("    keeping them apart (either as their own labels or by hand) will beat the MID merge.")

    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved: {out_dir}/")


if __name__ == "__main__":
    main()
