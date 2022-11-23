from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np

from .rect import Rect


@dataclass
class Chip:
    rect: Rect
    image: np.ndarray


def chip(image: np.ndarray, size_wh: Tuple[int, int], stride_xy: Tuple[int, int], full_chips: bool) -> List[List[Chip]]:
    """
    Chip an image into smaller images of size `size` with a stride of `stride`.

    Args:
        image: The image to chip (H, W, C).
        size_wh: The size of the chips (W, H).
        stride_xy: The stride of the chips (X, Y).
    """

    h, w, c = image.shape

    nw = (w - size_wh[0] + stride_xy[0] - 1) // stride_xy[0] + 1
    nh = (h - size_wh[1] + stride_xy[1] - 1) // stride_xy[1] + 1

    X = np.arange(nw) * stride_xy[0]
    Y = np.arange(nh) * stride_xy[1]

    chips: List[List[Chip]] = []
    coordinates: Set[Tuple[int, int]] = set()

    for y in Y:
        chips.append([])
        for x in X:
            if full_chips:
                # Attempt to to position the chip so it is not cut off at the bottom and right edges.
                x = max(0, min(x, w - size_wh[0]))
                y = max(0, min(y, h - size_wh[1]))

            if (x, y) in coordinates:
                continue
            coordinates.add((x, y))

            chip_data = image[y:y + size_wh[1], x:x + size_wh[0]]
            cw, ch = chip_data.shape[1], chip_data.shape[0]
            chips[-1].append(Chip(
                Rect.from_ltrb(x, y, x + cw, y + ch),
                chip_data
            ))

    return chips
