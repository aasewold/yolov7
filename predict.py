#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime
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
    return f'{id+1} {l} {t} {r} {b}'


def _eval_batch(batch: torch.Tensor, model: torch.nn.Module, params: Params) -> List[List[Label]]:
    pred_batch, _ = model(batch, augment=False)
    pred_batch = non_max_suppression(pred_batch, conf_thres=params.conf_threshold,
                                     iou_thres=params.iou_threshold, multi_label=True)

    batch_img_hw = batch.shape[-2:]
    batch_img_rect = Rect(0, 0, batch_img_hw[1], batch_img_hw[0])

    # (n, 6) x y x y conf cls
    labels_batch: List[List[Label]] = []
    for pred in pred_batch:
        out = list(map(tuple, pred.cpu().numpy()))
        out.sort(key=lambda x: x[4], reverse=True)
        out = out[:5]
        labels = [Label(int(cls), Rect.from_ltrb(l, t, r, b).intersection(batch_img_rect)) for (l, t, r, b, conf, cls) in out]
        labels_batch.append(labels)
    return labels_batch


@torch.inference_mode()
def _eval_images(dataset: ChippedDataset, model: torch.nn.Module, device: torch.device, params: Params):
    pbar = tqdm(total=dataset.num_images)
    dataloader = dataset.make_loader(params.num_workers)
    for batch, paths in dataloader:
        batch = batch.to(device)
        labels_batch = _eval_batch(batch, model, params)
        for path, labels in zip(paths, labels_batch):
            yield (path, labels)
        pbar.update(len(paths))


def _load_model(weight_paths: List[Path], device, trace=False) -> torch.nn.Module:
    model = attempt_load(weight_paths, map_location=device)
    if trace:
        model = TracedModel(model, device, 640)
    model.eval()
    return model


def _find_weights(path: Path, allowed_names: List[str]) -> List[Path]:
    collected_weight_paths: List[Path] = []
    candidates: List[Path] = []

    path = path.resolve()
    for entry in path.iterdir():
        if entry.is_dir():
            collected_weight_paths.extend(_find_weights(entry, allowed_names))
        elif entry.suffix == '.pt':
            candidates.append(entry)

    for name in allowed_names:
        chosen_candidate = next((c for c in candidates if c.name == name), None)
        if chosen_candidate:
            collected_weight_paths.append(chosen_candidate)
            break
    else:
        if candidates:
            print(f'WARNING: no allowed name found for {path.name}, adding all')
            collected_weight_paths.extend(candidates)

    return collected_weight_paths


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
    image_prefix: str = typer.Option('Norway', '--prefix'),
):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if dataset_path.is_file():
        dataset_yaml_dict = yaml.safe_load(dataset_path.read_text())
        if not split in dataset_yaml_dict:
            typer.echo(f'No "{split}" split in dataset YAML file')
            raise typer.Exit(1)
        data_path = Path(dataset_yaml_dict[split])
        default_predictions_path = f'{dataset_path.name}_{timestamp}.txt'
    else:
        data_path = dataset_path
        default_predictions_path = f'{"_".join(dataset_path.parts[-4:-2])}_{timestamp}.txt'
    
    if not data_path.exists():
        typer.echo(f"{dataset_path} doesn't exist.")
        raise typer.Exit(1)

    collected_weight_paths: List[Path] = []
    for weight_path in weight_paths:
        if weight_path.is_file():
            collected_weight_paths.append(weight_path)
        else:
            collected_weight_paths.extend(_find_weights(weight_path, ['last_stripped.pt', 'last.pt', 'best_stripped.pt', 'best.pt']))

    if not collected_weight_paths:
        typer.echo('No weight files found')
        raise typer.Exit(1)
    
    collected_weight_list = '\n'.join([str(path) for path in collected_weight_paths])
    print(f"Using weights:\n{collected_weight_list}")

    if not predictions_path:
        predictions_path = Path('output') / default_predictions_path

    if predictions_path.exists():
        typer.echo(f'Output path {predictions_path} already exists.')
        raise typer.Exit(1)

    device = select_device(device_str)

    params = Params(conf_threshold, iou_threshold, batch_size, num_workers)

    dataset = ChippedDataset(data_path, image_prefix, params.batch_size)
    model = _load_model(collected_weight_paths, device)
    
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with predictions_path.open('w') as f:
            predictions = _eval_images(dataset, model, device, params)
            for image_path, labels in predictions:
                assert len(labels) <= 5
                prediction_str = ' '.join(map(_format_label, labels))
                f.write(f'{image_path.name},{prediction_str}\n')
    except:
        if predictions_path.exists():
            predictions_path.unlink()
        raise


if __name__ == "__main__":
    typer.run(main)
