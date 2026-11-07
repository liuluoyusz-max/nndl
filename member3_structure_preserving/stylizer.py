"""Member 3 generator, built on member 2's unchanged AdaIN encoder/decoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from member2_adain.adain import AdaINModel, tensor_to_pil
from member3_structure_preserving.structure_map import (
    StructureConfig, build_structure_map, spatial_alpha,
)


@dataclass
class PreparedContent:
    image: torch.Tensor
    features: torch.Tensor
    structure: torch.Tensor


@dataclass
class StylizationResult:
    image: Image.Image
    content: Image.Image
    structure: torch.Tensor
    alpha_map: torch.Tensor
    feature_alpha: torch.Tensor
    baseline: Image.Image | None = None


class StructurePreservingStylizer:
    def __init__(self, model: AdaINModel | None = None,
                 structure_config: StructureConfig | None = None, **model_options):
        self.model = model if model is not None else AdaINModel(**model_options)
        self.structure_config = structure_config or StructureConfig()
        self._style_cache: dict[Path, torch.Tensor] = {}

    @torch.no_grad()
    def prepare_content(self, content) -> PreparedContent:
        image = self.model.prepare_image(content, self.model.content_size).to(self.model.device)
        self._validate_image(image)
        structure = build_structure_map(image, self.structure_config)
        return PreparedContent(image, self.model.encode(image), structure)

    @staticmethod
    def _validate_image(image):
        if image.shape[0] != 1 or min(image.shape[-2:]) < 32:
            raise ValueError("stylization requires one RGB image, each side at least 32 pixels")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("image values must be finite and in [0,1]")

    @torch.no_grad()
    def encode_style(self, style) -> torch.Tensor:
        key = Path(style).resolve() if isinstance(style, (str, Path)) else None
        if key is not None and key in self._style_cache:
            return self._style_cache[key]
        image = self.model.prepare_image(style, self.model.style_size).to(self.model.device)
        self._validate_image(image)
        features = self.model.encode(image)
        if key is not None:
            self._style_cache[key] = features
        return features

    @torch.no_grad()
    def render(self, content: PreparedContent, style_features: torch.Tensor,
               alpha: float = 0.5, beta: float = 0.7,
               with_baseline: bool = False) -> StylizationResult:
        alpha_map = spatial_alpha(content.structure, alpha, beta)
        # Area averaging preserves the contribution of narrow contours when
        # reducing the map to relu4_1 resolution. The equation is linear in M.
        feature_alpha = F.interpolate(alpha_map, size=content.features.shape[-2:], mode="area")
        mixed = self.model.mix(content.features, style_features, feature_alpha)
        image = tensor_to_pil(self.model.decode(mixed))
        baseline = None
        if with_baseline:
            baseline = tensor_to_pil(self.model.decode(
                self.model.mix(content.features, style_features, alpha)))
        return StylizationResult(image, tensor_to_pil(content.image),
                                 content.structure.cpu(), alpha_map.cpu(),
                                 feature_alpha.cpu(), baseline)

    def stylize(self, content, style, alpha=0.5, beta=0.7, with_baseline=False):
        return self.render(self.prepare_content(content), self.encode_style(style),
                           alpha, beta, with_baseline)


def stylize(content, style, alpha=0.5, beta=0.7,
            generator: StructurePreservingStylizer | None = None) -> Image.Image:
    """Single-image entry point; reuse a generator instance for many images."""
    generator = generator or StructurePreservingStylizer()
    return generator.stylize(content, style, alpha, beta).image
