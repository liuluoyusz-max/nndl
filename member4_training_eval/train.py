"""Unified training script for the three experimental groups.

Usage:
    python train.py --group original --seed 0
    python train.py --group adain  --seed 0 --aug-manifest augmented/adain/manifest_alpha0.5.txt

Training is step-budgeted: every group runs exactly TOTAL_STEPS optimizer
steps regardless of dataset size, so the three groups are comparable in
optimization budget (Section 3.7 of the report).  The reported model is the
*final* checkpoint of that budget: no checkpoint is selected on any validation
split, and target domains never touch training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import config
import data
from evaluate import evaluate_model
from model import build_model, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=config.GROUPS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--aug-manifest", type=Path, default=None,
                        help="stylization manifest for groups adain/ours")
    parser.add_argument("--tag", default=None, help="run tag; defaults to <group>_s<seed>")
    parser.add_argument("--out-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save step_XXX.pt checkpoints at this cadence "
                             "(used to trace accuracy vs. training budget)")
    parser.add_argument("--eval-targets", action="store_true",
                        help="evaluate target domains during training (diagnostic "
                             "only; never used for checkpoint selection)")
    args = parser.parse_args()

    tag = args.tag or f"{args.group}_s{args.seed}"
    out_dir = args.out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    train_ds = data.load_train_dataset(args.group, args.aug_manifest)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    print(f"[{tag}] train images: {len(train_ds)}  device: {config.DEVICE}")

    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LR,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.TOTAL_STEPS)

    val_ds = data.load_eval_dataset("photo_val")
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=2, pin_memory=True)

    target_loaders = {}
    if args.eval_targets:
        for split in ("art_painting_test", "cartoon_test", "sketch_test"):
            target_loaders[split] = DataLoader(
                data.load_eval_dataset(split), batch_size=128, shuffle=False,
                num_workers=2, pin_memory=True)

    history = []
    best = {"step": -1, "val_acc": -1.0}
    step = 0
    t0 = time.time()
    model.train()
    while step < config.TOTAL_STEPS:
        for images, labels in train_loader:
            if step >= config.TOTAL_STEPS:
                break
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.EVAL_EVERY == 0 or step == config.TOTAL_STEPS:
                val_acc = evaluate_model(model, val_loader)
                record = {"step": step, "loss": loss.item(), "val_acc": val_acc}
                marker = ""
                if val_acc > best["val_acc"]:
                    best = {"step": step, "val_acc": val_acc}
                    torch.save(model.state_dict(), out_dir / "best.pt")
                    marker = " *"
                if args.eval_targets:
                    for split in ("art_painting_test", "cartoon_test", "sketch_test"):
                        record[split] = evaluate_model(
                            model, target_loaders[split])
                if args.save_every and (step % args.save_every == 0 or step == config.TOTAL_STEPS):
                    torch.save(model.state_dict(), out_dir / f"step_{step:04d}.pt")
                history.append(record)
                extra = "".join(f" {k.replace('_test','')[:5]} {record[k]*100:5.2f}"
                                for k in ("art_painting_test", "cartoon_test", "sketch_test")
                                if k in record)
                print(f"[{tag}] step {step:4d}/{config.TOTAL_STEPS}  "
                      f"loss {loss.item():.4f}  val {val_acc*100:5.2f}%{marker}"
                      f"{extra}  ({time.time()-t0:.0f}s)")
                model.train()

    torch.save(model.state_dict(), out_dir / "last.pt")
    if config.CHECKPOINT == "last":
        # Report the final checkpoint (no selection bias); keep best.pt on disk
        # for reference and for the ablation analyses.
        summary_best = {"step": config.TOTAL_STEPS, "val_acc": best["val_acc"]}
    else:
        summary_best = best
    summary = {
        "group": args.group,
        "seed": args.seed,
        "tag": tag,
        "train_images": len(train_ds),
        "aug_manifest": str(args.aug_manifest) if args.aug_manifest else None,
        "config_hash": config.config_hash(),
        "training_config": config.training_config(),
        "best_val_acc": best["val_acc"],
        "best_step": best["step"],
        "reported_val_acc": summary_best["val_acc"],
        "reported_step": summary_best["step"],
        "total_steps": config.TOTAL_STEPS,
        "wall_time_sec": round(time.time() - t0, 1),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] done. best val {best['val_acc']*100:.2f}% @ step {best['step']}")


if __name__ == "__main__":
    main()
