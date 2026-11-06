"""Evaluate a trained checkpoint on all six splits and write results JSON.

Usage:
    python evaluate.py --run runs/original_s0 [--splits photo_val photo_test ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
import data
from model import build_model, set_seed

ALL_SPLITS = ("photo_val", "photo_test", "art_painting_test",
              "cartoon_test", "sketch_test")


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory with best.pt")
    parser.add_argument("--splits", nargs="+", default=list(ALL_SPLITS))
    parser.add_argument("--checkpoint", default=f"{config.CHECKPOINT}.pt")
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    set_seed(0)
    model = build_model()
    state = torch.load(args.run / args.checkpoint, map_location=config.DEVICE,
                       weights_only=True)
    model.load_state_dict(state)

    results = {"run": str(args.run), "checkpoint": args.checkpoint,
               "config_hash": config.config_hash()}
    for split in args.splits:
        loader = DataLoader(data.load_eval_dataset(split), batch_size=128,
                            shuffle=False, num_workers=2, pin_memory=True)
        acc = evaluate_model(model, loader)
        results[split] = acc
        print(f"{split:20s} {acc*100:6.2f}%")

    cross = [results.get(s) for s in ("art_painting_test", "cartoon_test", "sketch_test")]
    if all(a is not None for a in cross):
        results["cross_domain_avg"] = sum(cross) / 3
        print(f"{'cross-domain avg':20s} {results['cross_domain_avg']*100:6.2f}%")

    out = Path(args.run) / f"eval_{Path(args.checkpoint).stem}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
