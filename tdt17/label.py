from dataclasses import dataclass

from .rect import Rect


@dataclass
class Label:
    id: int
    rect: Rect

    @classmethod
    def from_xywh_string(cls, string: str) -> 'Label':
        id, x, y, w, h = string.strip().split()
        return cls(int(id), Rect.from_xywh(float(x), float(y), float(w), float(h)))
    
    @classmethod
    def from_ltrb_string(cls, string: str) -> 'Label':
        id, l, t, r, b = string.strip().split()
        return cls(int(id), Rect.from_ltrb(float(l), float(t), float(r), float(b)))

    def __str__(self) -> str:
        x, y, w, h = self.rect.xywh
        return f'{self.id} {x} {y} {w} {h}'
    
    def __repr__(self) -> str:
        return f'Label({self.id}, {self.rect})'
