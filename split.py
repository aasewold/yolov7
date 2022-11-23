from pathlib import Path
import random

def main(
    dataset: Path,
    output: Path,
    num_validation: int = 1024,
    val_prefix: str = 'Norway'
):
    """Split a dataset into train and validation sets."""
    dataset = dataset.resolve()
    output = output.resolve()
    if not dataset.is_dir():
        raise ValueError(f'Dataset {dataset} does not exist')
    
    output.mkdir(exist_ok=False)
    (output / 'images').mkdir()
    (output / 'labels').mkdir()
    (output / 'images' / 'train').mkdir()
    (output / 'images' / 'val').mkdir()
    (output / 'labels' / 'train').mkdir()
    (output / 'labels' / 'val').mkdir()

    input_images = [ p.resolve() for p in (dataset / 'images').iterdir() if p.is_file() ]
    input_labels = [ p.resolve() for p in (dataset / 'labels').iterdir() if p.is_file() ]

    candidate_images = [ p for p in input_images if p.name.startswith(val_prefix) ]

    percentage = num_validation / len(candidate_images)
    print(f'Using {num_validation} of {len(candidate_images)} images for validation ({percentage:.1%})')
    val_images = random.sample(candidate_images, num_validation)
    val_image_stems = { p.stem for p in val_images }
    val_labels = [ p for p in input_labels if p.stem in val_image_stems ]

    train_images = [ p for p in input_images if p.stem not in val_image_stems ]
    train_labels = [ p for p in input_labels if p.stem not in val_image_stems ]

    for p in val_images:
        Path(output / 'images' / 'val' / p.name).symlink_to(p)
    for p in val_labels:
        Path(output / 'labels' / 'val' / p.name).symlink_to(p)
    for p in train_images:
        Path(output / 'images' / 'train' / p.name).symlink_to(p)
    for p in train_labels:
        Path(output / 'labels' / 'train' / p.name).symlink_to(p)
    

if __name__ == '__main__':
    import typer
    typer.run(main)
