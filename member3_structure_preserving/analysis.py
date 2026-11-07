"""Diagnostic panels and the same 224px Canny overlap used by member 4."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from member3_structure_preserving.stylizer import StylizationResult


def map_image(values: torch.Tensor) -> Image.Image:
    """Absolute [0,1] grayscale scale; alpha panels are not auto-normalized."""
    array = values.detach().cpu().squeeze().clamp(0, 1).mul(255).round().numpy().astype(np.uint8)
    return Image.fromarray(array)


def comparison_panel(result: StylizationResult, style_path: Path, alpha: float, beta: float) -> Image.Image:
    with Image.open(style_path) as source:
        style = source.convert("RGB")
    panels = [("Content", result.content), ("Style", style),
              ("Structure M [0,1]", map_image(result.structure)),
              (f"Alpha [0,1], base={alpha:g}", map_image(result.alpha_map))]
    if result.baseline is not None:
        panels.append(("Standard AdaIN", result.baseline))
    panels.append((f"Ours, beta={beta:g}", result.image))
    width, height = 224, 254
    canvas = Image.new("RGB", (width * len(panels), height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, picture) in enumerate(panels):
        draw.text((i * width + 6, 6), label, fill="black")
        picture = ImageOps.contain(picture.convert("RGB"), (width - 8, height - 30))
        canvas.paste(picture, (i * width + (width - picture.width) // 2,
                               28 + (height - 30 - picture.height) // 2))
    return canvas


class StructureMetrics:
    def __init__(self, with_lpips=False, device="cpu"):
        # Fail before generation if an explicitly requested dependency is absent.
        import cv2
        self.cv2 = cv2
        self.device = device
        self.net = None
        if with_lpips:
            import lpips
            self.net = lpips.LPIPS(net="alex").to(device).eval()

    @staticmethod
    def _rgb(path):
        with Image.open(path) as source:
            return source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    def compare(self, content_path: Path, generated_path: Path):
        a, b = self._rgb(content_path), self._rgb(generated_path)
        # Match member 4: grayscale before resize, with fixed Canny 100/200.
        def edges(path):
            with Image.open(path) as source:
                gray = np.asarray(source.convert("L").resize((224, 224), Image.Resampling.BILINEAR))
            return self.cv2.Canny(gray, 100, 200) > 0
        ea, eb = edges(content_path), edges(generated_path)
        na, nb = int(ea.sum()), int(eb.sum())
        score = float(2 * np.logical_and(ea, eb).sum() / (na + nb)) if na and nb else None
        result = {"edge_score": score, "lpips": None}
        if self.net is not None:
            def tensor(image):
                values = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
                return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                result["lpips"] = float(self.net(tensor(a), tensor(b)).item())
        return result


def summarize(records: list[dict]) -> dict:
    result = {}
    for key in ("edge_score", "lpips", "mean_alpha"):
        values = [r[key] for r in records if r.get(key) is not None and math.isfinite(r[key])]
        result[key] = {"mean": float(np.mean(values)),
                       "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                       "n": len(values)} if values else None
    return result
