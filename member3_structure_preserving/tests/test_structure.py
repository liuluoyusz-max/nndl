from __future__ import annotations

import unittest

import torch

from member3_structure_preserving.structure_map import (
    StructureConfig, build_structure_map, spatial_alpha,
)
from member3_structure_preserving.stylizer import StructurePreservingStylizer
from member4_training_eval import config


class StructureMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flat_images_have_no_structure(self):
        for value in (0, 0.3, 1):
            response = build_structure_map(torch.full((2, 3, 33, 65), float(value)))
            self.assertEqual(response.shape, (2, 1, 33, 65))
            self.assertEqual(float(response.max()), 0)

    def test_step_edge_is_protected_and_beta_is_monotone(self):
        image = torch.zeros(1, 3, 64, 96)
        image[:, :, :, 48:] = 1
        response = build_structure_map(image)
        self.assertGreater(float(response[..., 44:52].mean()), 0.3)
        self.assertEqual(float(response[..., :12].max()), 0)
        mild = spatial_alpha(response, 0.5, 0.3)
        strong = spatial_alpha(response, 0.5, 1)
        self.assertTrue(torch.all(strong <= mild))
        self.assertGreaterEqual(float(strong.min()), 0)
        self.assertLessEqual(float(mild.max()), 0.5)

    def test_sparse_edges_survive_quantile_normalization(self):
        image = torch.zeros(1, 3, 256, 256)
        image[:, :, 127:129, 127:129] = 1
        self.assertGreater(float(build_structure_map(image).max()), 0.05)

    def test_invalid_and_nonfinite_inputs_are_rejected(self):
        for image in (torch.zeros(1, 1, 64, 64), torch.full((3, 32, 32), float("nan")),
                      torch.full((3, 32, 32), -0.1)):
            with self.assertRaises(ValueError):
                build_structure_map(image)
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                spatial_alpha(torch.zeros(1, 1, 2, 2), beta=value)
        with self.assertRaises(ValueError):
            StructureConfig(edge_quantile=0)


class PretrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        weights = config.PROJECT_ROOT / "member2_adain" / "models"
        if not all((weights / name).is_file() for name in ("decoder.pth", "vgg_normalised.pth")):
            raise unittest.SkipTest("pretrained weights not present locally")
        cls.generator = StructurePreservingStylizer(content_size=64, style_size=64,
                                                   device="cpu", weights_dir=weights)

    def test_beta_zero_matches_member2_at_odd_aspect_ratio(self):
        torch.manual_seed(1)
        content, style = torch.rand(1, 3, 65, 91), torch.rand(1, 3, 64, 80)
        result = self.generator.stylize(content, style, alpha=0.5, beta=0, with_baseline=True)
        original = self.generator.model.stylize(content, style, alpha=0.5)
        self.assertEqual(result.image.tobytes(), original.tobytes())
        self.assertEqual(result.image.tobytes(), result.baseline.tobytes())
        self.assertEqual(result.image.size, (96, 72))
        self.assertTrue(torch.all(result.feature_alpha == 0.5))

    def test_zero_alpha_is_decoder_reconstruction_not_original(self):
        torch.manual_seed(2)
        content = torch.rand(1, 3, 64, 64)
        style = torch.rand_like(content)
        result = self.generator.stylize(content, style, alpha=0, beta=1)
        reference = self.generator.model.stylize(content, style, alpha=0)
        self.assertEqual(result.image.tobytes(), reference.tobytes())
        self.assertEqual(float(result.alpha_map.max()), 0)

    def test_spatial_modulation_changes_feature_strength(self):
        image = torch.zeros(1, 3, 64, 64)
        image[..., 32:] = 1
        result = self.generator.stylize(image, torch.rand_like(image), beta=1)
        self.assertLess(float(result.feature_alpha.min()), 0.5)
        self.assertTrue(torch.isfinite(result.feature_alpha).all())
        self.assertEqual(result.structure.shape[-2:], (64, 64))


if __name__ == "__main__":
    unittest.main()
