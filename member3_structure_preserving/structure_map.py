"""Soft edge response M and alpha = alpha0 * (1 - beta * M).

No target images, labels, or learned segmentation models enter this map.
All operations use PyTorch, so OpenCV is not required for generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureConfig:
    preblur_sigma: float = 1.0
    edge_quantile: float = 0.95
    dilation_radius: int = 1
    map_blur_sigma: float = 2.0

    def __post_init__(self):
        for name in ("preblur_sigma", "map_blur_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError(f"{name} must be finite and in [0, 10]")
        if not math.isfinite(self.edge_quantile) or not 0 < self.edge_quantile <= 1:
            raise ValueError("edge_quantile must be in (0, 1]")
        if not isinstance(self.dilation_radius, int) or not 0 <= self.dilation_radius <= 16:
            raise ValueError("dilation_radius must be an integer in [0, 16]")

    def to_dict(self):
        return asdict(self)


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return image
    radius = max(1, math.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (axis / sigma).square())
    kernel /= kernel.sum()
    # Replicate also handles images smaller than the Gaussian kernel.
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                     kernel.reshape(1, 1, 1, -1))
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"),
                    kernel.reshape(1, 1, -1, 1))


@torch.no_grad()
def build_structure_map(image: torch.Tensor, config: StructureConfig | None = None) -> torch.Tensor:
    """RGB CHW/BCHW in [0,1] -> B1HW soft structural response in [0,1]."""
    config = config or StructureConfig()
    image = torch.as_tensor(image).detach().float()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1:
        raise ValueError("image must be a non-empty RGB CHW or BCHW tensor")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image must contain finite RGB values in [0,1]")
    weights = image.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    gray = _blur((image * weights).sum(dim=1, keepdim=True), config.preblur_sigma)
    sx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((sx, sx.T)).unsqueeze(1)
    gradient = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kernels)
    magnitude = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    # Do not amplify round-off on constant images into a spurious edge map.
    magnitude = torch.where(magnitude > 1e-6, magnitude, 0.0)
    flat = magnitude.flatten(1)
    scale = torch.quantile(flat, config.edge_quantile, dim=1)
    # Sparse contours may occupy less than 5% of an image.
    scale = torch.where(scale > 1e-6, scale, flat.amax(dim=1)).clamp_min(1e-6)
    response = (magnitude / scale[:, None, None, None]).clamp(0, 1)
    radius = config.dilation_radius
    if radius:
        response = F.max_pool2d(response, 2 * radius + 1, stride=1, padding=radius)
    return _blur(response, config.map_blur_sigma).clamp(0, 1)


def spatial_alpha(structure: torch.Tensor, alpha: float = 0.5, beta: float = 0.7) -> torch.Tensor:
    """Keep more content near strong structure; beta=0 is constant AdaIN."""
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0,1]")
    if structure.numel() == 0 or not torch.isfinite(structure).all():
        raise ValueError("structure must be non-empty and finite")
    if structure.min() < 0 or structure.max() > 1:
        raise ValueError("structure values must be in [0,1]")
    return alpha * (1 - beta * structure)
