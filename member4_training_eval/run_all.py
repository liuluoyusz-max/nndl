"""Run one configuration across all seeds, then evaluate.

Usage:
    python run_all.py --group original
    python run_all.py --group adain --aug-manifest <path>
    python run_all.py --group ours  --aug-manifest <path> --beta 0.5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=config.GROUPS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    parser.add_argument("--aug-manifest", type=Path, default=None)
    parser.add_argument("--tag-prefix", default=None)
    args = parser.parse_args()

    per_seed = []
    for seed in args.seeds:
        tag = f"{args.tag_prefix or args.group}_s{seed}"
        run_dir = config.OUTPUT_ROOT / tag
        eval_file = run_dir / f"eval_{config.CHECKPOINT}.json"
        if not eval_file.exists():
            print(f"=== {tag}: training ===")
            cmd = [sys.executable, "train.py", "--group", args.group,
                   "--seed", str(seed), "--tag", tag]
            if args.aug_manifest:
                cmd += ["--aug-manifest", str(args.aug_manifest)]
            subprocess.run(cmd, check=True, cwd=config.CODE_ROOT)
            print(f"=== {tag}: evaluating ===")
            subprocess.run([sys.executable, "evaluate.py", "--run", str(run_dir),
                            "--checkpoint", f"{config.CHECKPOINT}.pt"],
                           check=True, cwd=config.CODE_ROOT)
        else:
            print(f"=== {tag}: eval already present, skipping ===")
        per_seed.append(json.loads(eval_file.read_text()))

    summary = {"group": args.group, "seeds": args.seeds, "config_hash": config.config_hash()}
    for split in ("photo_test",) + CROSS:
        values = [r[split] for r in per_seed if split in r]
        if values:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
            summary[split] = {"mean": mean, "std": var ** 0.5, "per_seed": values}
    cross_means = [s[split]["mean"] for split in CROSS if split in summary]
    if len(cross_means) == 3:
        summary["cross_domain_avg"] = {"mean": sum(cross_means) / 3}

    out = config.OUTPUT_ROOT / f"summary_{args.tag_prefix or args.group}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
