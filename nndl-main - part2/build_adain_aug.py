"""Build the Photo-training AdaIN augmentation set and its manifest.

Outputs, relative to the project folder:

    data/aug_adain/images/photo/<class>/<file>.jpg
    data/aug_adain/photo_train.txt
    data/aug_adain/generation.json

Member 4 can read the set with PACSDataset(data_root='data/aug_adain',
split_file='data/aug_adain/photo_train.txt'). Only photo_train.txt is augmented.
"""
import argparse
import json
from pathlib import Path

from adain_paths import (
    output_relative,
    project_root,
    read_split,
    resolve_data_root,
    resolve_split_file,
)
from style_pool import StylePool


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=None, help='Folder that contains images/. Defaults to data/ or pacs/.')
    parser.add_argument('--split-file', type=Path, default=None)
    parser.add_argument('--style-dir', type=Path, default=root / 'style_pool')
    parser.add_argument('--output-dir', type=Path, default=root / 'data' / 'aug_adain')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--styles-per-image', type=int, default=1)
    parser.add_argument('--content-size', type=int, default=512)
    parser.add_argument('--style-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='Generate only the first N outputs into output-dir/preview.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--weights-dir', type=Path, default=root / 'models')
    parser.add_argument('--allow-pacs-styles', action='store_true')
    parser.add_argument('--allow-other-split', action='store_true')
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error('--alpha must be between 0 and 1')
    if args.styles_per_image < 1:
        parser.error('--styles-per-image must be positive')
    if args.limit < 0:
        parser.error('--limit must be non-negative')

    data_root = resolve_data_root(root, args.data_root)
    split_file = resolve_split_file(root, args.split_file, args.allow_other_split)
    image_root = data_root / 'images'
    samples = read_split(split_file, image_root)
    if split_file.name == 'photo_train.txt' and len(samples) != 1349:
        print(f'Warning: expected 1349 Photo training images, found {len(samples)}.')
    pool = StylePool(args.style_dir, protected_root=image_root, allow_pacs_styles=args.allow_pacs_styles)
    assigned = pool.assign(len(samples), args.styles_per_image, args.seed)
    multiple = args.styles_per_image > 1
    jobs = []
    for (relative, label, content_path), styles in zip(samples, assigned):
        for style_index, style_path in enumerate(styles):
            jobs.append((relative, label, content_path, style_path, style_index))
    seen = set()
    for relative, _label, _content_path, style_path, style_index in jobs:
        relative_out = output_relative(relative, style_path, style_index, multiple)
        if relative_out in seen:
            raise ValueError(f'Duplicate augmented path: {relative_out}')
        seen.add(relative_out)

    output_dir = args.output_dir
    manifest_name = 'photo_train.txt'
    if args.limit:
        output_dir = output_dir / 'preview'
        manifest_name = 'photo_train_partial.txt'
        jobs = jobs[:args.limit]
    image_out = output_dir / 'images'
    style_root = pool.style_dir.resolve()
    settings = {
        'split': split_file.name,
        'alpha': args.alpha,
        'seed': args.seed,
        'content_size': args.content_size,
        'style_size': args.style_size,
        'styles_per_image': args.styles_per_image,
        'limit': args.limit,
        'count': len(jobs),
        'styles': [path.relative_to(style_root).as_posix() for path in pool.paths],
    }
    _check_settings(output_dir / 'generation.json', settings, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'generation.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')

    from adain import AdaINModel

    model = AdaINModel(weights_dir=args.weights_dir, content_size=args.content_size, style_size=args.style_size)
    if model.device.type == 'cpu' and len(jobs) > 20:
        print(f'CPU generation of {len(jobs)} images at size {args.content_size} will take a long time.')
    lines = []
    pairs = []
    for number, (relative, label, content_path, style_path, style_index) in enumerate(jobs, 1):
        relative_out = output_relative(relative, style_path, style_index, multiple)
        destination = image_out / relative_out
        if destination.is_file() and destination.stat().st_size > 0 and not args.overwrite:
            print(f'[{number}/{len(jobs)}] skip {relative_out}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            model.stylize(content_path, style_path, args.alpha).save(destination, quality=95)
            print(f'[{number}/{len(jobs)}] wrote {relative_out}')
        lines.append(f'{relative_out} {label}')
        pairs.append(f'{relative_out}\t{relative}\t{style_path.relative_to(style_root).as_posix()}\t{label}')
    manifest = output_dir / manifest_name
    manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output_dir / 'pairs.txt').write_text('\n'.join(pairs) + '\n', encoding='utf-8')
    data_root_text = output_dir.resolve().relative_to(root).as_posix()
    manifest_text = manifest.resolve().relative_to(root).as_posix()
    print(f'Augmented images: {len(lines)}')
    print(f'Manifest: {manifest}')
    print(f"Load with PACSDataset(data_root='{data_root_text}', split_file='{manifest_text}').")


def _check_settings(path, settings, overwrite):
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding='utf-8'))
    comparable = {key: value for key, value in settings.items() if key != 'count'}
    old = {key: value for key, value in previous.items() if key != 'count'}
    if comparable != old and not overwrite:
        raise SystemExit(
            f'Existing generation settings in {path} do not match this run. '
            'Use the same alpha, seed, sizes, and style pool, or pass --overwrite.'
        )


if __name__ == '__main__':
    main()
