from dataclasses import dataclass

from .rect import Rect


@dataclass
class Label:
    name: str
    rect: Rect

    @classmethod
    def from_string(cls, string: str) -> 'Label':
        name, x, y, w, h = string.strip().split()
        return cls(name, Rect.from_xywh(float(x), float(y), float(w), float(h)))
    
    def int(self) -> 'Label':
        return Label(self.name, self.rect.int())

    def __str__(self) -> str:
        x, y, w, h = self.rect.xywh
        return f'{self.name} {x} {y} {w} {h}'
