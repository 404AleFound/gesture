import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

GESTURES = ("LEFT_THUMB", "RIGHT_THUMB", "LEFT_INDEX", "RIGHT_INDEX")
DIRECTIONS = ("RIGHT", "LEFT", "DOWN", "UP")

# iPhone 15 Pro logical resolution used to preserve on-screen aspect ratio.
IPHONE_ASPECT = 2556 / 1179


@dataclass
class PathPoint:
    t: int
    x: float
    y: float


@dataclass
class Scroll:
    type: str
    tag: str
    start: int
    end: int
    count: int
    path: list[PathPoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Scroll":
        return cls(
            type=data["type"],
            tag=data["tag"],
            start=data["start"],
            end=data["end"],
            count=data["count"],
            path=[PathPoint(**p) for p in data.get("path", [])],
        )

    @property
    def duration_ms(self) -> int:
        return self.end - self.start

    @property
    def direction(self) -> str:
        """Infer direction from first->last displacement.

        Screen coordinates: +x right, +y down. Dominant axis (larger |delta|)
        decides horizontal vs vertical, its sign decides which side.
        """
        if len(self.path) < 2:
            return "UP"
        dx = self.path[-1].x - self.path[0].x
        dy = self.path[-1].y - self.path[0].y
        if abs(dx) >= abs(dy):
            return "RIGHT" if dx >= 0 else "LEFT"
        return "DOWN" if dy >= 0 else "UP"

    @property
    def full_tag(self) -> str:
        """gesture+direction combined label, e.g. 'LEFT_THUMB_UP'."""
        return f"{self.tag}_{self.direction}"

    def __repr__(self) -> str:
        return (
            f"Scroll(tag={self.tag!r}, dir={self.direction}, "
            f"duration={self.duration_ms}ms, points={self.count})"
        )


def load_scrolls(jsonl_path: str | Path) -> list[Scroll]:
    jsonl_path = Path(jsonl_path)
    scrolls: list[Scroll] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip {jsonl_path.name}:{line_no} ({e})", file=sys.stderr)
                continue
            scrolls.append(Scroll.from_dict(data))
    return scrolls


def _setup_phone_axes(ax, title: str) -> None:
    """Common axes config: normalized 0..1, y inverted, phone aspect ratio."""
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect(IPHONE_ASPECT)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)


def _draw_scroll(ax, scroll: "Scroll", color, label: str | None) -> None:
    """Plot a single scroll onto ax with start/end markers."""
    if not scroll.path:
        return
    xs = [p.x for p in scroll.path]
    ys = [p.y for p in scroll.path]
    ax.plot(xs, ys, color=color, alpha=0.55, linewidth=1, label=label)
    ax.scatter([xs[0]], [ys[0]], color=color, s=15, marker="o")
    ax.scatter([xs[-1]], [ys[-1]], color=color, s=15, marker="x")


def _color_map(keys, cmap_name: str = "tab10") -> dict:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    cmap = plt.get_cmap(cmap_name)
    return {k: cmap(i % 10) for i, k in enumerate(keys)}


def plot_all(
    scrolls: list[Scroll],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """One figure with all scrolls, colored by gesture tag."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    color_by_tag = _color_map(GESTURES)
    fig, ax = plt.subplots(figsize=(6, 6 * IPHONE_ASPECT / 2))
    seen: set[str] = set()
    for scroll in scrolls:
        label = scroll.tag if scroll.tag not in seen else None
        seen.add(scroll.tag)
        _draw_scroll(ax, scroll, color_by_tag[scroll.tag], label)
    _setup_phone_axes(ax, f"All scrolls (n={len(scrolls)})")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_by_gesture(
    scrolls: list[Scroll],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """1x4 subplots, one per gesture; direction colors within each subplot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    color_by_dir = _color_map(DIRECTIONS)
    fig, axes = plt.subplots(2, 2, figsize=(7, 11))
    for ax, gesture in zip(axes.flat, GESTURES):
        subset = [s for s in scrolls if s.tag == gesture]
        seen: set[str] = set()
        for scroll in subset:
            direction = scroll.direction
            label = direction if direction not in seen else None
            seen.add(direction)
            _draw_scroll(ax, scroll, color_by_dir[direction], label)
        _setup_phone_axes(ax, f"{gesture} (n={len(subset)})")
        if seen:
            ax.legend(loc="lower right", fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.05, wspace=0.15, hspace=0.15)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_by_direction(
    scrolls: list[Scroll],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """1x4 subplots, one per direction; gesture colors within each subplot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    color_by_tag = _color_map(GESTURES)
    fig, axes = plt.subplots(2, 2, figsize=(7, 11))
    for ax, direction in zip(axes.flat, DIRECTIONS):
        subset = [s for s in scrolls if s.direction == direction]
        seen: set[str] = set()
        for scroll in subset:
            label = scroll.tag if scroll.tag not in seen else None
            seen.add(scroll.tag)
            _draw_scroll(ax, scroll, color_by_tag[scroll.tag], label)
        _setup_phone_axes(ax, f"{direction} (n={len(subset)})")
        if seen:
            ax.legend(loc="lower right", fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.05, wspace=0.15, hspace=0.15)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_combined(
    scrolls: list[Scroll],
    out_dir: str | Path,
    show: bool = False,
) -> None:
    """4 figures (one per gesture), each with 1x4 subplots per direction.

    Each subplot shows only the (gesture, direction) subset, single color.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. run: pip install matplotlib", file=sys.stderr)
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    color_by_tag = _color_map(GESTURES)

    for gesture in GESTURES:
        fig, axes = plt.subplots(2, 2, figsize=(7, 11))
        color = color_by_tag[gesture]
        for ax, direction in zip(axes.flat, DIRECTIONS):
            subset = [
                s for s in scrolls if s.tag == gesture and s.direction == direction
            ]
            for scroll in subset:
                _draw_scroll(ax, scroll, color, None)
            _setup_phone_axes(ax, f"{gesture} + {direction} (n={len(subset)})")

        fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.05, wspace=0.15, hspace=0.15)
        fig.savefig(out_dir / f"combined_{gesture}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def plot_scrolls_all_views(
    scrolls: list[Scroll],
    out_dir: str | Path = "figs/scrolls",
    show: bool = False,
) -> None:
    """Render all four visualization views into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all(scrolls, save_path=out_dir / "all.png", show=show)
    plot_by_gesture(scrolls, save_path=out_dir / "by_gesture.png", show=show)
    plot_by_direction(scrolls, save_path=out_dir / "by_direction.png", show=show)
    plot_combined(scrolls, out_dir=out_dir, show=show)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scrolls.py <file.jsonl>", file=sys.stderr)
        sys.exit(1)
    jsonl_path = sys.argv[1]
    scrolls = load_scrolls(jsonl_path)
    print(f"loaded {len(scrolls)} scrolls from {jsonl_path}")
    for s in scrolls[:5]:
        print(s)
    plot_scrolls_all_views(scrolls)
