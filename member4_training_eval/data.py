"""Datasets for the three experimental groups.

All groups read through the same dataset classes so that the only difference
between runs is the list of images, never the loading code.  Original images
live under ``DATA_ROOT/PACS`` and stylized images under ``DATA_ROOT/augmented``
(both relative to the same root), so one dataset class serves both.

Augmented images enter training through a *manifest* written by the
generators (members 2 and 3), one manifest per generator:

    <path under DATA_ROOT> <label 1..7> <original-relative-path> <style-name> <alpha> [beta]

The first two fields mirror member 1's split format; the trailing fields are
provenance metadata recorded for fairness checks and later analysis.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import config


class ManifestDataset(Dataset):
    """Images from (relative_path, label) pairs under one root directory."""

    def __init__(self, rows, image_root, transform):
        self.image_root = Path(image_root)
        self.transform = transform
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rel, label = self.rows[index]
        with Image.open(self.image_root / rel) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_eval(split_name):
    """Rows for one of the six evaluation splits, relative to DATA_ROOT."""
    return [(f"PACS/{rel}", label) for rel, label in config.read_split(config.SPLIT_FILES[split_name])]


def load_eval_dataset(split_name):
    return ManifestDataset(load_eval(split_name), config.DATA_ROOT, config.EVAL_TRANSFORM)


def read_aug_manifest(path):
    """Parse a stylization manifest with provenance fields."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rel, raw_label = parts[0], int(parts[1])
        meta = parts[2:]                      # original, style, alpha, [beta]
        rows.append((rel.replace("\\", "/"), raw_label - 1, meta))
    return rows


def load_train_rows(group, aug_manifest=None):
    """Training rows for one experimental group, relative to DATA_ROOT.

    original: photo_train split only.
    adain/ours: photo_train + (path, label) rows from the group manifest.
    """
    rows = [(f"PACS/{rel}", label) for rel, label in config.read_split(config.SPLIT_FILES["photo_train"])]
    if group in ("adain", "ours"):
        if aug_manifest is None:
            raise ValueError(f"group '{group}' requires an augmentation manifest")
        for rel, label, _meta in read_aug_manifest(aug_manifest):
            rows.append((rel, label))
    elif group != "original":
        raise ValueError(f"unknown group: {group}")
    return rows


def load_train_dataset(group, aug_manifest=None):
    return ManifestDataset(load_train_rows(group, aug_manifest), config.DATA_ROOT, config.TRAIN_TRANSFORM)
