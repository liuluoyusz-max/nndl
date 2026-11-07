"""Read an external AdaIN style-image pool.

Style images must stay outside the PACS image folder. Art, Cartoon, and Sketch
are evaluation domains, so using them as styles leaks the test appearance into
training.
"""
import random
from pathlib import Path

from adain_paths import IMAGE_SUFFIXES


class StylePool:
    def __init__(self, style_dir, protected_root=None, allow_pacs_styles=False):
        self.style_dir = Path(style_dir)
        if not self.style_dir.is_dir():
            raise FileNotFoundError(
                f'Style pool not found: {self.style_dir}. Create this folder and put style images in it.'
            )
        protected = Path(protected_root).resolve() if protected_root is not None else None
        self.paths = []
        for path in sorted(self.style_dir.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            if protected is not None and not allow_pacs_styles and _is_inside(resolved, protected):
                raise ValueError(
                    f'Style image is inside the PACS image folder: {resolved}. '
                    'Put external style images in the style pool. '
                    'Art, Cartoon, and Sketch test images cannot be used as styles.'
                )
            self.paths.append(resolved)
        if not self.paths:
            raise ValueError(
                f'No style images found in {self.style_dir}. '
                'Add jpg or png files. Do not use the PACS test domains.'
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.paths[index]

    def assign(self, count, styles_per_image, seed):
        """Return a style list for each content image. The assignment depends only on seed and order."""
        if styles_per_image < 1:
            raise ValueError('styles_per_image must be positive')
        if styles_per_image > len(self.paths):
            raise ValueError(
                f'styles_per_image={styles_per_image} exceeds the style pool size {len(self.paths)}'
            )
        order = list(range(len(self.paths)))
        random.Random(seed).shuffle(order)
        assigned = []
        pool_size = len(order)
        for index in range(count):
            start = (index * styles_per_image) % pool_size
            chosen = [self.paths[order[(start + offset) % pool_size]] for offset in range(styles_per_image)]
            assigned.append(chosen)
        return assigned


def _is_inside(path, root):
    return path == root or root in path.parents
