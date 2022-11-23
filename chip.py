#!/usr/bin/env python3
import multiprocessing
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import typer
import yaml
from tqdm import tqdm

import tdt17.chip
from tdt17.label import Label


@dataclass
class Params:
    num_cpus: int
    chip_size: Tuple[int, int]
    chip_stride: Tuple[int, int]
    label_coverage_threshold: float
    full_chips: bool


def _get_labels(chip: tdt17.chip.Chip, labels: List[Label], image_size: Tuple[int, int], params: Params) -> List[str]:
    w, h = image_size
    chip_labels = []

    for label in labels:
        label_rect = label.rect.scale(w, h)
        intersection = label_rect.intersection(chip.rect)
        coverage = intersection.area / label_rect.area
        if coverage >= params.label_coverage_threshold:
            intersection = intersection.offset(-chip.rect.l, -chip.rect.t)
            intersection = intersection.scale(1 / chip.rect.w, 1 / chip.rect.h)
            chip_labels.append(Label(label.id, intersection))

    return chip_labels


def _chip_image(
    in_images: Path, in_labels: Path,
    out_images: Path, out_labels: Path,
    params: Params,
    name: str,
):
    in_image = in_images / name
    in_label = in_labels / (in_image.stem + '.txt')

    out_image_base = out_images / name
    out_label_base = out_labels / (out_image_base.stem + '.txt')

    if in_label.exists():
        labels = list(map(Label.from_xywh_string, in_label.read_text().splitlines()))
    else:
        labels = []

    image = cv2.imread(str(in_image), cv2.IMREAD_COLOR)
    image_wh = image.shape[1], image.shape[0]
    chips = tdt17.chip.chip(image, params.chip_size, params.chip_stride, params.full_chips)

    for row, chip_row in enumerate(chips):
        for col, chip in enumerate(chip_row):
            out_image = out_image_base.with_name(f'{out_image_base.stem}_{row}_{col}{out_image_base.suffix}')
            out_label = out_label_base.with_name(f'{out_label_base.stem}_{row}_{col}{out_label_base.suffix}')

            cv2.imwrite(str(out_image), chip.image)

            chip_labels = _get_labels(chip, labels, image_wh, params)
            if chip_labels:
                out_label.write_text('\n'.join(map(str, chip_labels)) + '\n')


class Chipper:
    def __init__(self, in_images: Path, in_labels: Path, out_images: Path, out_labels: Path, params: Params):
        self.in_images = in_images
        self.in_labels = in_labels
        self.out_images = out_images
        self.out_labels = out_labels
        self.params = params

    def __call__(self, name: str):
        _chip_image(self.in_images, self.in_labels,
                    self.out_images, self.out_labels,
                    self.params, name)


def _chip_folder(in_dataset: Path, out_dataset: Path, split: str, params: Params):
    in_images = in_dataset / 'images' / split
    in_labels = in_dataset / 'labels' / split
    out_images = out_dataset / 'images' / split
    out_labels = out_dataset / 'labels' / split

    image_names = [
        image.name
        for image in in_images.iterdir()
        if image.is_file()
    ]

    out_images.mkdir(parents=True, exist_ok=False)
    out_labels.mkdir(parents=True, exist_ok=False)

    chipper = Chipper(in_images, in_labels, out_images, out_labels, params)

    if params.num_cpus > 1:
        with multiprocessing.Pool(params.num_cpus) as pool:
            iterable = pool.imap_unordered(chipper, image_names)
            for _ in tqdm(iterable, total=len(image_names)):
                pass
    else:
        for name in tqdm(image_names):
            chipper(name)


def _make_yaml_config(yaml_path: Path, yaml_dict: dict, output_path: Path, splits: List[str]):
    yaml_dict = dict(yaml_dict)
    for split in splits:
        yaml_dict[split] = str(output_path / 'images' / split)
    output_yaml = yaml.dump(yaml_dict, sort_keys=False)
    yaml_path.write_text(output_yaml)


def main(
    dataset_path: Path = typer.Argument(..., metavar='dataset', exists=True),
    size: Tuple[int, int] = typer.Option((640, 640), '--size'),
    stride: Tuple[int, int] = typer.Option((320, 320), '--stride'),
    coverage_threshold: float = typer.Option(0.25, '--threshold'),
    output_path: Optional[Path] = typer.Option(None, '--output'),
    output_path_suffix: str = typer.Option('', '--suffix'),
    no_full_chips: bool = typer.Option(False, '--no-full'),
    num_cpus: int = typer.Option(0, '--cpus'),
    overwrite: bool = typer.Option(False),
    splits: List[str] = typer.Option(..., '--split'),
):

    # Parse dataset YAML description (or don't)

    if dataset_path.is_file():
        dataset_yaml_dict = yaml.safe_load(dataset_path.read_text())
        dataset_yaml_paths = {Path(dataset_yaml_dict[split]).parent.parent for split in splits}
        if len(dataset_yaml_paths) != 1:
            raise ValueError('All splits must be in the same dataset')
        dataset_path = dataset_yaml_paths.pop()
    else:
        dataset_yaml_dict = {
            'nc': 4,
            'names': ['D00', 'D10', 'D20', 'D40'],
        }

    # Check that stuff exists or doesn't

    if not (
        (dataset_path / 'images').exists()
        and (dataset_path / 'labels').exists()
    ):
        typer.echo('Invalid dataset (missing ./images and ./labels)')
        raise typer.Exit(1)

    if not output_path_suffix:
        output_path_suffix = f'-chipped-{size[0]}x{size[1]}-{stride[0]}x{stride[1]}-t{coverage_threshold}'

    if not output_path:
        output_path = dataset_path.with_name(dataset_path.stem + output_path_suffix)

    output_path = output_path.resolve()

    typer.echo(f'Chipping to {output_path}')

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            typer.echo(f'Output path {output_path} already exists. Use --overwrite to overwrite it.')
            raise typer.Exit(1)

    if num_cpus == 0:
        num_cpus = multiprocessing.cpu_count()

    # Split the splits

    params = Params(num_cpus, size, stride, coverage_threshold, full_chips=not no_full_chips)

    for split in splits:
        typer.echo(f'Chipping {split} split')
        _chip_folder(dataset_path, output_path, split, params)

    # Output some information

    for split in splits:
        dataset_yaml_dict[split] = str(output_path / 'images' / split)
    dataset_yaml_dict['chip'] = asdict(params)
    output_yaml_path = output_path / 'dataset.yaml'
    output_yaml_path.write_text(yaml.safe_dump(dataset_yaml_dict, sort_keys=False))


if __name__ == '__main__':
    typer.run(main)
