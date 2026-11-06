"""Save a few AdaIN examples: content, style, and stylized result side by side."""
import argparse
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
    parser.add_argument('--output-dir', type=Path, default=root / 'data' / 'aug_adain' / 'samples')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--content-size', type=int, default=512)
    parser.add_argument('--style-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights-dir', type=Path, default=root / 'models')
    parser.add_argument('--allow-pacs-styles', action='store_true')
    parser.add_argument('--allow-other-split', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    if not 0.0 <= args.alpha <= 1.0:
        parser.error('--alpha must be between 0 and 1')

    data_root = resolve_data_root(root, args.data_root)
    split_file = resolve_split_file(root, args.split_file, args.allow_other_split)
    samples = read_split(split_file, data_root / 'images')
    pool = StylePool(args.style_dir, protected_root=data_root / 'images', allow_pacs_styles=args.allow_pacs_styles)
    chosen = samples[:args.num_samples]
    assigned = pool.assign(len(samples), styles_per_image=1, seed=args.seed)[:len(chosen)]

    from adain import AdaINModel

    model = AdaINModel(weights_dir=args.weights_dir, content_size=args.content_size, style_size=args.style_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for (relative, _label, content_path), styles in zip(chosen, assigned):
        style_path = styles[0]
        result = model.stylize(content_path, style_path, args.alpha)
        name = Path(output_relative(relative, style_path, 0, multiple_styles=True)).name
        result_path = args.output_dir / name
        result.save(result_path, quality=95)
        comparison = _comparison(content_path, style_path, result)
        comparison_path = args.output_dir / f'{result_path.stem}__compare.jpg'
        comparison.save(comparison_path, quality=95)
        records.append(f'{relative}\t{style_path.name}\t{result_path.name}\t{args.alpha}')
        print(f'Saved {result_path.name}')
    manifest = args.output_dir / 'samples.txt'
    manifest.write_text('\n'.join(records) + '\n', encoding='utf-8')
    print(f'Saved {len(records)} samples to {args.output_dir}')
    print(f'Device: {model.device}. These previews are not the training augmentation set.')


def _open_rgb(path):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert('RGB')
        image.load()
        return image


def _comparison(content_path, style_path, result):
    from PIL import Image

    panels = []
    for image in (_open_rgb(content_path), _open_rgb(style_path), result):
        ratio = 256 / image.height
        panels.append(image.resize((max(1, round(image.width * ratio)), 256), Image.Resampling.BILINEAR))
    canvas = Image.new('RGB', (sum(panel.width for panel in panels), 256))
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    return canvas


if __name__ == '__main__':
    main()
