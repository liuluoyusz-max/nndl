"""PyTorch Dataset for PACS text manifests: relative/image/path label_1_to_7."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
KNOWN_EXCLUDED = {'sketch/dog/n02103406_4068-1.png'}


class PACSDataset(Dataset):
    def __init__(self, data_root, split_file, transform=None):
        self.image_root = (Path(data_root) / 'images').resolve()
        self.transform = transform
        self.samples = []
        seen = set()
        for line in Path(split_file).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            relative, raw_label = line.rsplit(maxsplit=1)
            relative = relative.replace('\\', '/').removeprefix('./')
            if relative in KNOWN_EXCLUDED:
                continue
            path = (self.image_root / relative).resolve()
            label = int(raw_label) - 1
            if self.image_root not in path.parents or not path.is_file():
                raise ValueError(f'Invalid image path: {relative}')
            if not 0 <= label < len(CLASSES) or path.parent.name != CLASSES[label]:
                raise ValueError(f'Invalid class label: {line}')
            if path in seen:
                raise ValueError(f'Duplicate sample: {relative}')
            seen.add(path)
            self.samples.append((path, label))
        if not self.samples:
            raise ValueError('The split contains no usable samples.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
