from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from PIL import Image
from tqdm import tqdm


class ChippedDataset(IterableDataset[Tuple[torch.Tensor, Path]]):
    def __init__(self, root_path: Path, filter_prefix: str, batch_size: int) -> None:
        super().__init__()
        self._batch_size = batch_size

        self._all_paths = [
            image
            for image in root_path.iterdir()
            if image.is_file() and image.name.startswith(filter_prefix)
        ]
        self._paths = _group_by_size(self._all_paths)
        self._batches = _make_batches(self._paths, batch_size)
    
    @property
    def num_images(self) -> int:
        return len(self._all_paths)
    
    def __len__(self) -> int:
        return len(self._batches)
    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info:
            start = worker_info.id
            stride = worker_info.num_workers
        else:
            start = 0
            stride = 1
        for i in range(start, len(self._batches), stride):
            paths = self._batches[i]
            img = torch.stack([self._load_image(path) for path in paths])
            yield img, paths
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        img_np = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_np = img_np[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to CHW
        img_np = np.ascontiguousarray(img_np)

        img = torch.from_numpy(img_np)
        img = _pad_mod(img)
        img = img.float()
        img /= 255.0

        assert img.shape[-2] % 32 == 0, 'height must be a multiple of 32'
        assert img.shape[-1] % 32 == 0, 'width must be a multiple of 32'
        return img
    
    def make_loader(self, num_workers: int, **kwargs) -> DataLoader:
        return DataLoader(
            self,
            pin_memory=True,
            batch_size=None,
            num_workers=num_workers,
            **kwargs,
        )


def _group_by_size(paths: List[Path]):
    sizes: Dict[Tuple[int, int], List[Path]] = {}
    for path in tqdm(paths, desc="Grouping images by size"):
        img = Image.open(path)
        img_width, img_height = img.size
        padded_width, padded_height = _calc_size(img_width, img_height, 32)
        sizes.setdefault((padded_width, padded_height), []).append(path)
    return list(sizes.items())


def _make_batches(grouped_paths: List[Tuple[Tuple[int, int], List[Path]]], batch_size: int):
    batches = []
    for _, paths in grouped_paths:
        num_batches = len(paths) // batch_size
        for i in range(num_batches):
            batches.append(paths[i * batch_size : (i + 1) * batch_size])
        if len(paths) % batch_size != 0:
            batches.append(paths[num_batches * batch_size :])
    return batches


def _calc_size(img_width: int, img_height: int, stride: int) -> Tuple[int, int]:
    w_pad = (stride - img_width % stride) % stride
    h_pad = (stride - img_height % stride) % stride
    return img_width + w_pad, img_height + h_pad


def _pad_mod(img: torch.Tensor, stride: int = 32) -> torch.Tensor:
    img_height, img_width = img.shape[-2:]
    padded_width, padded_height = _calc_size(img_width, img_height, stride)
    dw = padded_width - img_width
    dh = padded_height - img_height
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    return F.pad(img, [left, top, right, bottom], fill=114)
