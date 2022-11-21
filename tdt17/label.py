from dataclasses import dataclass

from .rect import Rect


@dataclass
class Label:
    id: int
    rect: Rect

    @classmethod
    def from_string(cls, string: str) -> 'Label':
        id, x, y, w, h = string.strip().split()
        return cls(int(id), Rect.from_xywh(float(x), float(y), float(w), float(h)))
    
    def int(self) -> 'Label':
        return Label(self.id, self.rect.int())

    def __str__(self) -> str:
        x, y, w, h = self.rect.xywh
        return f'{self.id} {x} {y} {w} {h}'
