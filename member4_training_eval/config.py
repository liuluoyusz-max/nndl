"""Central configuration for all training and evaluation runs.

Every quantity that the three experimental groups (Original / +AdaIN / +Ours)
must share lives here.  The config hash computed by ``config_hash()`` is
stored in every result file and checked by ``fairness_check.py`` so that a
run with a mismatched configuration cannot silently enter the results table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torchvision import transforms

# ---------------------------------------------------------------- paths ----
CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
import os
DATA_ROOT = Path(os.environ.get(
    "NNDL_DATA_ROOT", PROJECT_ROOT / "data"))
# Shared local layout: <project>/data/PACS. NNDL_DATA_ROOT overrides data/.
IMAGE_ROOT = DATA_ROOT / "PACS"          # member 1's layout: images under data/PACS
SPLIT_DIR = CODE_ROOT / "project_splits"
OUTPUT_ROOT = PROJECT_ROOT / "runs"
STYLE_POOL_DIR = CODE_ROOT / "style_pool"
STYLE_POOL_IMAGES = STYLE_POOL_DIR / "images"
STYLE_POOL_MANIFEST = STYLE_POOL_DIR / "style_pool.json"
AUG_ROOT = DATA_ROOT / "augmented"       # stylized images written by members 2/3

# ------------------------------------------------------------- protocol ----
# Member 1's label mapping, 0-based: dog=0, elephant=1, giraffe=2, guitar=3,
# horse=4, house=5, person=6.  verify_splits.py enforces folder/label
# consistency, so an order change here fails loud.
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
GROUPS = ("original", "adain", "ours")
SPLIT_FILES = {
    "photo_train": SPLIT_DIR / "photo_train.txt",
    "photo_val": SPLIT_DIR / "photo_val.txt",
    "photo_test": SPLIT_DIR / "photo_test.txt",
    "art_painting_test": SPLIT_DIR / "art_painting_test.txt",
    "cartoon_test": SPLIT_DIR / "cartoon_test.txt",
    "sketch_test": SPLIT_DIR / "sketch_test.txt",
}

SEEDS = (0, 1, 2)

# ----------------------------------------------------------- optimization ---
BATCH_SIZE = 64
NUM_WORKERS = 4
# Same optimizer-step budget for every group: 30 nominal epochs over the
# 1,349-image Original training set at batch size 64 (30 * ceil(1349/64)).
ORIGINAL_EPOCHS = 30
TOTAL_STEPS = 660
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
EVAL_EVERY = 60            # photo-val evaluation cadence in optimizer steps
# Checkpoint policy.  Photo Validation accuracy saturates at 97-98% within the
# first 60 steps (150 images => 1 image = 0.67%), so selecting the "best
# validation" checkpoint picks noise and more than doubles the seed-to-seed
# spread of cross-domain accuracy (std 7.22 vs 2.71 in our baseline).  We
# therefore use the final checkpoint of the fixed step budget: it introduces no
# selection bias at all and is the protocol reported in Sections 3.3 and 3.7.
CHECKPOINT = "last"        # "last" | "best"
# Human-readable description of the checkpoint policy, recorded in every run
# summary.  It must agree with CHECKPOINT: "last" means the final checkpoint
# of the fixed step budget, with no validation-based selection.
CHECKPOINT_SELECTION = "final checkpoint of the fixed step budget"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ aug ----
# One stylized image per original training image, for both augmented groups.
AUG_PER_IMAGE = 1
# Size of the frozen external style pool.  The pool holds paintings that are
# NOT part of PACS (see style_pool/SOURCES.txt); PACS Art Painting, Cartoon
# and Sketch are evaluation domains and must never be used as style sources.
N_STYLE_IMAGES = 10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Geometric augmentation only: photometric augmentation is intentionally
# excluded because style statistics are the variable under study.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def read_split(path: Path) -> list[tuple[Path, int]]:
    """Parse a member-1 split manifest: ``relative/path label_1_to_7``."""
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rel, raw = line.rsplit(maxsplit=1)
        rows.append((rel.replace("\\", "/").removeprefix("./"), int(raw) - 1))
    return rows


def training_config() -> dict:
    """The fairness-relevant subset of the configuration.

    ``checkpoint_selection`` is documentation of the policy, not a behaviour:
    it is recorded in every run summary but excluded from ``config_hash()`` so
    that rewording it never invalidates runs that are already on disk.
    """
    return {
        "batch_size": BATCH_SIZE,
        "total_steps": TOTAL_STEPS,
        "lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "eval_every": EVAL_EVERY,
        "checkpoint": CHECKPOINT,
        "train_transform": repr(TRAIN_TRANSFORM),
        "eval_transform": repr(EVAL_TRANSFORM),
        "arch": "resnet18",
        "pretrained": "imagenet",
        "checkpoint_selection": CHECKPOINT_SELECTION,
    }


# The fields that decide what the trained model is.  This is an explicit
# allow-list rather than "everything except", so that adding a descriptive
# field to training_config() can never silently change the fingerprint.
#
# ``checkpoint`` and ``checkpoint_selection`` are deliberately NOT in it:
# they describe *which saved checkpoint we report*, not how the model is
# trained, and the choice is applied identically to every group by the same
# code path.  They are still recorded in every run summary, so the reported
# policy stays auditable.
_BEHAVIOURAL_FIELDS = (
    "batch_size",
    "total_steps",
    "lr",
    "momentum",
    "weight_decay",
    "label_smoothing",
    "eval_every",
    "train_transform",
    "eval_transform",
    "arch",
    "pretrained",
)


def config_hash_of(training_config_dict: dict) -> str:
    """Canonical fingerprint of a recorded training configuration.

    Computed from the allow-list above, so it can be recomputed from the
    ``training_config`` block stored inside any run summary -- including runs
    recorded before this definition existed -- and compared with the current
    one.
    """
    behavioural = {k: training_config_dict.get(k) for k in _BEHAVIOURAL_FIELDS}
    payload = json.dumps(behavioural, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def config_hash() -> str:
    """Fingerprint of the behaviourally relevant configuration.

    Two runs may be compared only if this matches.
    """
    return config_hash_of(training_config())
