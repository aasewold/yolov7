#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import typer
import yaml
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


def _format_label(label: Label) -> str:
    id = label.id
    l, t, r, b = map(int, label.rect.ltrb)
    return f'{id} {l} {t} {r} {b}'


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


def _eval_images(dataset: ChippedDataset, model: torch.nn.Module, device: torch.device, params: Params):
    pbar = tqdm(total=dataset.num_images)
    dataloader = dataset.make_loader(params.num_workers)
    for batch, paths in dataloader:
        batch = batch.to(device)
        pred_batch = _eval_batch(batch, model, params)
        for pred, path in zip(pred_batch, paths):
            labels = [Label(int(cls), Rect.from_ltrb(l, t, r, b)) for (l, t, r, b, conf, cls) in pred]
            yield (path, labels)
        pbar.update(len(paths))


def _load_model(weight_paths: List[Path], device, trace=False) -> torch.nn.Module:
    model = attempt_load(weight_paths, map_location=device)
    if trace:
        model = TracedModel(model, device, 640)
    model.eval()
    return model


def main(
    dataset_path: Path = typer.Argument(..., exists=True),
    weight_paths: List[Path] = typer.Option(..., '-w', '--weights', exists=True),
    predictions_path: Optional[Path] = typer.Option(None, '-o', '--output', dir_okay=False),
    device_str: str = typer.Option('', '-d', '--device'),
    conf_threshold: float = typer.Option(0.25, '--conf'),
    iou_threshold: float = typer.Option(0.45, '--iou'),
    batch_size: int = typer.Option(32, '--batch-size', '--bs'),
    num_workers: int = typer.Option(24, '--num-workers', '--nw'),
    split: str = typer.Option('test', '--split'),
    overwrite: bool = typer.Option(False),
    image_prefix: str = typer.Option('Norway', '--prefix'),
):
    if dataset_path.is_file():
        dataset_yaml_dict = yaml.safe_load(dataset_path.read_text())
        if not split in dataset_yaml_dict:
            typer.echo(f'No "{split}" split in dataset YAML file')
            raise typer.Exit(1)
        data_path = Path(dataset_yaml_dict[split])
        default_predictions_path = dataset_path.name + '.txt'
    else:
        data_path = dataset_path
        default_predictions_path = '_'.join(dataset_path.parts[-4:-2]) + '.txt'
    
    if not data_path.exists():
        typer.echo(f"{dataset_path} doesn't exist.")
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
        predictions_path = Path('output') / default_predictions_path

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
    
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open('w') as f:
        predictions = _eval_images(dataset, model, device, params)
        for image_path, labels in predictions:
            assert len(labels) <= 5
            prediction_str = ' '.join(map(_format_label, labels))
            f.write(f'{image_path.name},{prediction_str}\n')


if __name__ == "__main__":
    typer.run(main)
