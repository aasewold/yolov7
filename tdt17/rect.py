from dataclasses import dataclass
from typing import Tuple


@dataclass
class Rect:
    l: float
    t: float
    r: float
    b: float

    def __post_init__(self):
        assert self.l <= self.r and self.t <= self.b

    @staticmethod
    def from_ltrb(l: float, t: float, r: float, b: float) -> 'Rect':
        return Rect(l, t, r, b)

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> 'Rect':
        return Rect.from_ltrb(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    @property
    def ltrb(self) -> Tuple[float, float, float, float]:
        return self.l, self.t, self.r, self.b

    @property
    def x(self) -> float:
        return self.l + self.w / 2

    @property
    def y(self) -> float:
        return self.t + self.h / 2

    @property
    def w(self) -> float:
        return self.r - self.l

    @property
    def h(self) -> float:
        return self.b - self.t

    @property
    def area(self) -> float:
        return self.w * self.h

    def __eq__(self, other: 'Rect') -> bool:
        return self.l == other.l and self.r == other.r and self.t == other.t and self.b == other.b

    def __bool__(self) -> bool:
        return self.area > 0

    def offset(self, x: float, y: float) -> 'Rect':
        return Rect.from_ltrb(self.l + x, self.t + y, self.r + x, self.b + y)

    def scale(self, x: float, y: float) -> 'Rect':
        return Rect.from_ltrb(self.l * x, self.t * y, self.r * x, self.b * y)

    def intersection(self, other: 'Rect') -> 'Rect':
        l = max(self.l, other.l)
        r = min(self.r, other.r)
        t = max(self.t, other.t)
        b = min(self.b, other.b)

        r = max(l, r)
        b = max(t, b)
        return Rect.from_ltrb(l, t, r, b)

    def iou(self, other: 'Rect') -> float:
        intersection = self.intersection(other)
        return intersection.area / (self.area + other.area - intersection.area)
