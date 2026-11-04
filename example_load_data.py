"""Read a PACS batch; --check-files validates paths without PyTorch."""
import argparse
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=project_root / 'data')
    parser.add_argument('--split-dir', type=Path, default=project_root / 'project_splits')
    parser.add_argument('--split', choices=['photo_train', 'photo_val', 'photo_test',
                        'art_painting_test', 'cartoon_test', 'sketch_test'], default='photo_train')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--check-files', action='store_true', help='Check paths and labels only; no PyTorch batch')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    image_root = (args.data_root / 'images').resolve()
    split_file = (args.split_dir / f'{args.split}.txt').resolve()
    if not image_root.is_dir():
        parser.error(f'Image folder not found: {image_root}. Set --data-root to the folder containing images.')
    if not split_file.is_file():
        parser.error(f'Split file not found: {split_file}. Set --split-dir to project_splits.')

    if args.check_files:
        classes = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
        seen = set()
        for number, line in enumerate(split_file.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                relative, raw_label = line.rsplit(maxsplit=1)
                label = int(raw_label) - 1
            except ValueError:
                parser.error(f'Invalid split row {number}: {line}')
            path = (image_root / relative.replace('\\', '/')).resolve()
            if image_root not in path.parents or not path.is_file():
                parser.error(f'Invalid/missing image at row {number}: {path}')
            if not 0 <= label < len(classes) or path.parent.name != classes[label]:
                parser.error(f'Label mismatch at row {number}: {line}')
            if path in seen:
                parser.error(f'Duplicate image: {path}')
            seen.add(path)
        if not seen:
            parser.error('Empty split')
        print(f'PASS: {args.split}: {len(seen)} valid image paths and labels.')
        print('This checks paths and labels only, not image decoding or a PyTorch batch.')
        return

    try:
        from torchvision import transforms
        from torch.utils.data import DataLoader
        from pacs_dataset import PACSDataset, CLASSES
    except ImportError as error:
        parser.exit(1, f'Missing dependency: {error}. Use a Python environment with torch, torchvision and Pillow installed.\n')

    # Reading smoke test only. Member 4 should define the final training transforms.
    check_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = PACSDataset(data_root=args.data_root, split_file=split_file,
                          transform=check_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    print(f'Split: {args.split}; samples: {len(dataset)}')
    print(f'Images shape: {tuple(images.shape)}; labels shape: {tuple(labels.shape)}')
    print(f'First label: {labels[0].item()} ({CLASSES[labels[0].item()]})')
    print('Read one batch successfully. No model training or accuracy evaluation performed.')


if __name__ == '__main__':
    main()
