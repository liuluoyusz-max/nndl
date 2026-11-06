"""Paths and Photo-training manifest helpers for the AdaIN augmentation set."""
from pathlib import Path

from pacs_dataset import CLASSES, KNOWN_EXCLUDED

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
AUG_ROOT = Path('data') / 'aug_adain'


def project_root():
    return Path(__file__).resolve().parent


def resolve_data_root(root, data_root):
    """Return the folder that contains images/. Prefer data/, then the downloaded pacs/."""
    if data_root is not None:
        chosen = Path(data_root)
    elif (root / 'data' / 'images').is_dir():
        chosen = root / 'data'
    elif (root / 'pacs' / 'images').is_dir():
        chosen = root / 'pacs'
    else:
        chosen = root / 'data'
    images = (chosen / 'images').resolve()
    if not images.is_dir():
        raise FileNotFoundError(
            f'Image folder not found: {images}. Put PACS images in data/images or pacs/images.'
        )
    return chosen.resolve()


def resolve_split_file(root, split_file, allow_other_split=False):
    path = Path(split_file) if split_file is not None else root / 'project_splits' / 'photo_train.txt'
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Split file not found: {path}')
    if path.name != 'photo_train.txt' and not allow_other_split:
        raise ValueError(
            f'Refusing split {path.name}. Member 2 augments photo_train.txt only.'
        )
    return path


def read_split(split_file, image_root):
    """Return (relative posix path, label 1-7, absolute image path) in manifest order."""
    image_root = Path(image_root).resolve()
    samples = []
    seen = set()
    for number, line in enumerate(Path(split_file).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            relative, raw_label = line.rsplit(maxsplit=1)
            label = int(raw_label)
        except ValueError:
            raise ValueError(f'Invalid split row {number}: {line}') from None
        relative = relative.replace('\\', '/').removeprefix('./')
        if relative in KNOWN_EXCLUDED:
            continue
        path = (image_root / relative).resolve()
        if image_root not in path.parents or not path.is_file():
            raise ValueError(f'Invalid image path at row {number}: {relative}')
        if not 1 <= label <= len(CLASSES) or path.parent.name != CLASSES[label - 1]:
            raise ValueError(f'Invalid class label at row {number}: {line}')
        if relative in seen:
            raise ValueError(f'Duplicate sample: {relative}')
        seen.add(relative)
        samples.append((relative, label, path))
    if not samples:
        raise ValueError('The split contains no usable samples.')
    return samples


def output_relative(relative, style_path, style_index, multiple_styles):
    """Manifest path under the augmented images/ folder."""
    source = Path(relative)
    if not multiple_styles:
        return source.with_suffix('.jpg').as_posix()
    style_stem = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in style_path.stem)[:40]
    name = f'{source.stem}__{style_index:02d}_{style_stem}.jpg'
    return (source.parent / name).as_posix()
