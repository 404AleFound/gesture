import json
import logging
from dataclasses import dataclass, field
from pathlib import Path


_log = logging.getLogger(__name__)


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
    direction: str
    tag: str
    start: int
    end: int
    count: int
    path: list[PathPoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Scroll":
        return cls(
            type=data["type"],
            direction=data["direction"].upper(),
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
                _log.warning("skip %s:%d (%s)", jsonl_path.name, line_no, e)
                continue
            scrolls.append(Scroll.from_dict(data))
    return scrolls


