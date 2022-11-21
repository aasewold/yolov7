#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
import typer
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.experimental import attempt_load
from tdt17.dataloader import ChippedDataset
from tdt17.label import Label
from tdt17.rect import Rect
from utils.general import non_max_suppression
from utils.torch_utils import TracedModel, select_device

Prediction = Tuple[float, float, float, float, float, int]


@dataclass
class Params:
    conf_threshold: float
    iou_threshold: float
    batch_size: int
    num_workers: int


@torch.inference_mode()
def _eval_batch(batch: torch.Tensor, model: torch.nn.Module, params: Params) -> List[List[Prediction]]:
    pred_batch, _ = model(batch, augment=False)
    pred_batch = non_max_suppression(pred_batch, conf_thres=params.conf_threshold,
                                     iou_thres=params.iou_threshold, multi_label=True)

    # (n, 6) x y x y conf cls
    out_batch: List[List[Prediction]] = []
    for pred in pred_batch:
        out = list(map(tuple, pred.cpu().numpy()))
        out.sort(key=lambda x: x[4], reverse=True)
        out_batch.append(out[:5])
    return out_batch


def _eval_images(dataset: ChippedDataset, model: torch.nn.Module, device: torch.device, result_writer: Callable[[Path, List[Label]], None], params: Params):
    dataloader = dataset.make_loader(params.num_workers)

    pbar = tqdm(total=dataset.num_images)
    for batch, paths in dataloader:
        batch = batch.to(device)
        pred_batch = _eval_batch(batch, model, params)
        for pred, path in zip(pred_batch, paths):
            labels = [Label(int(cls), Rect.from_ltrb(l, t, r, b)) for (l, t, r, b, conf, cls) in pred]
            result_writer(path, labels)
        pbar.update(len(paths))


def _load_model(weight_paths: List[Path], device, trace=False) -> torch.nn.Module:
    model = attempt_load(weight_paths, map_location=device)
    if trace:
        model = TracedModel(model, device, 640)
    model.eval()
    return model


def main(
    dataset_yaml_path: Path = typer.Argument(..., exists=True),
    weight_paths: List[Path] = typer.Option(..., '-w', '--weights', exists=True),
    predictions_path: Path = typer.Option(..., '-o', '--output', dir_okay=False),
    device_str: str = typer.Option('', '-d', '--device'),
    conf_threshold: float = typer.Option(0.25, '--conf'),
    iou_threshold: float = typer.Option(0.45, '--iou'),
    batch_size: int = typer.Option(32, '--batch-size', '--bs'),
    num_workers: int = typer.Option(24, '--num-workers', '--nw'),
    overwrite: bool = typer.Option(False),
    image_prefix: str = typer.Option('Norway', '--prefix'),
):
    dataset_yaml_dict = yaml.safe_load(dataset_yaml_path.read_text())

    if not 'test' in dataset_yaml_dict:
        typer.echo('No test split in dataset YAML file')
        raise typer.Exit(1)

    data_path = Path(dataset_yaml_dict['test'])
    if not data_path.exists():
        typer.echo(f'No test set found in {dataset_yaml_path}')
        raise typer.Exit(1)

    collected_weight_paths: List[Path] = []
    for path in weight_paths:
        if path.is_dir():
            collected_weight_paths.extend(path.glob('*.pt'))
        else:
            collected_weight_paths.append(path)

    if not collected_weight_paths:
        typer.echo('No weight files found')
        raise typer.Exit(1)

    if not predictions_path:
        predictions_path = Path('output/predictions.txt')

    if predictions_path.exists():
        if overwrite:
            predictions_path.unlink()
        else:
            typer.echo(f'Output path {predictions_path} already exists. Use --overwrite to overwrite it.')
            raise typer.Exit(1)

    device = select_device(device_str)

    params = Params(conf_threshold, iou_threshold, batch_size, num_workers)

    dataset = ChippedDataset(data_path, image_prefix, params.batch_size)
    model = _load_model(collected_weight_paths, device)
    
    predictions_path.mkdir(parents=True, exist_ok=True)
    with predictions_path.open('w') as f:
        def result_writer(image_path: Path, labels: List[Label]):
            assert len(labels) <= 5
            prediction_str = ' '.join(map(str, labels))
            f.write(f'{image_path.stem} {prediction_str}\n')

        _eval_images(dataset, model, device, result_writer, params)


if __name__ == "__main__":
    typer.run(main)
