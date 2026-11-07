from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from member4_training_eval import config
from member3_structure_preserving.analysis import StructureMetrics
from member3_structure_preserving.generate import parser, run
from member3_structure_preserving.protocol import FrozenProtocol, contained, sha256


class ProtocolAndPipelineTests(unittest.TestCase):
    def setUp(self):
        weights = config.PROJECT_ROOT / "member2_adain" / "models"
        if not all((weights / name).is_file() for name in ("decoder.pth", "vgg_normalised.pth")):
            self.skipTest("local pretrained weights required")
        self.temporary = tempfile.TemporaryDirectory(prefix="nndl-member3-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.rows = config.read_split(config.SPLIT_FILES["photo_train"])
        # Synthetic fixtures use protocol filenames ONLY in an isolated temp
        # tree. They are never copied into the user's PACS directory.
        for i, (relative, _) in enumerate(self.rows[:2]):
            path = self.root / "PACS" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (80, 64), (30, 50, 90))
            draw = ImageDraw.Draw(image)
            draw.rectangle((15 + i * 5, 10, 65, 50), fill=(220, 180, 60))
            draw.line((5, 58, 75, 5), fill="white", width=3)
            image.save(path)

    def arguments(self, *extra):
        return parser().parse_args(["build", "--data-root", str(self.root), "--limit", "2",
                                   "--content-size", "64", "--style-size", "64",
                                   "--threads", "2", "--with-adain", "--diagnostics", "--metrics", *extra])

    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(args)

    def test_manifests_are_accepted_by_member4_and_resume_repairs_corruption(self):
        summaries = self.invoke(self.arguments())
        manifest = Path(summaries[0]["manifest"])
        self.assertTrue(manifest.is_relative_to(self.root / "aug_sp"))
        self.assertFalse((self.root / "augmented" / "manifest_ours.txt").exists())
        lines = manifest.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line.split()) == 6 for line in lines))
        self.assertTrue((manifest.parent / "photo_train_partial.txt").is_file())
        self.assertFalse((manifest.parent / "photo_train.txt").exists())
        paths = [self.root / line.split()[0] for line in lines]
        hashes = [sha256(path) for path in paths]
        times = [path.stat().st_mtime_ns for path in paths]
        self.invoke(self.arguments())
        self.assertEqual(times, [path.stat().st_mtime_ns for path in paths])
        paths[0].write_bytes(b"interrupted/corrupt image")
        self.invoke(self.arguments())
        self.assertEqual(hashes, [sha256(path) for path in paths])
        # Wrong generation settings must fail without withdrawing valid manifests.
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.invoke(self.arguments("--preblur-sigma", "1.5"))
        self.assertTrue(manifest.is_file())
        env = dict(os.environ, NNDL_DATA_ROOT=str(self.root))
        result = subprocess.run([
            sys.executable, str(config.CODE_ROOT / "fairness_check.py"), "--subset",
            "--adain", str(manifest.parent / "manifest_adain.txt"), "--ours", str(manifest)],
            env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
        # Exercise the actual member-4 reader, including 1-based -> 0-based labels.
        reader = subprocess.run([
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import data; "
            "rows=data.read_aug_manifest(sys.argv[2]); "
            "assert len(rows)==2 and all(r[1]==0 and len(r[2])==4 for r in rows); "
            "ds=data.load_train_dataset('ours',sys.argv[2]); assert len(ds)==1351; "
            "image,label=ds[1349]; assert image.shape==(3,224,224) and label==0",
            str(config.CODE_ROOT), str(manifest)], env=env, text=True, capture_output=True)
        self.assertEqual(reader.returncode, 0, reader.stdout + reader.stderr)

    def test_full_generation_refuses_missing_sources_and_frozen_pairing_tampering(self):
        with self.assertRaises(FileNotFoundError):
            self.invoke(self.arguments("--limit", "0"))
        self.assertFalse((self.root / "augmented").exists())
        self.assertFalse((self.root / "aug_sp").exists())
        manifest = json.loads(config.STYLE_POOL_MANIFEST.read_text())
        original = next(iter(manifest["pairing"]))
        manifest["pairing"]["cartoon/dog/test.jpg"] = manifest["pairing"].pop(original)
        altered = self.root / "tampered_pool.json"
        altered.write_text(json.dumps(manifest))
        with patch.object(config, "STYLE_POOL_MANIFEST", altered):
            with self.assertRaisesRegex(ValueError, "coverage differ"):
                FrozenProtocol(self.root)
        for path in ("../outside.jpg", str(self.root / "inside.jpg")):
            with self.assertRaises(ValueError):
                contained(self.root, path)

    def test_balanced_samples_cover_all_classes_with_frozen_pairing(self):
        for label in range(len(config.CLASSES)):
            relative = next(rel for rel, value in self.rows if value == label)
            path = self.root / "PACS" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), "gray").save(path)
        protocol = FrozenProtocol(self.root)
        jobs = protocol.jobs(7, balanced=True)
        self.assertEqual({job.label for job in jobs}, set(range(1, 8)))
        self.assertTrue(all(protocol.manifest["pairing"][job.original] == job.style for job in jobs))

    def test_official_layout_exports_both_manifests_after_preview(self):
        preview = self.root / "aug_sp" / "previews" / "existing"
        preview.mkdir(parents=True)
        marker = preview / "keep.txt"
        marker.write_text("preserve this preview")
        # Reduce only this isolated fixture protocol to two source images so
        # the test exercises official publishing without creating 1349 files.
        protocol = FrozenProtocol(self.root)
        protocol.rows = protocol.rows[:2]
        with patch("member3_structure_preserving.generate.FrozenProtocol", return_value=protocol):
            self.invoke(self.arguments("--limit", "0"))
        root = self.root / "aug_sp"
        primary = root / "manifest_ours.txt"
        bridge = self.root / "augmented" / "manifest_ours.txt"
        self.assertEqual(primary.read_bytes(), bridge.read_bytes())
        self.assertTrue(all(row.split()[0].startswith("aug_sp/images/")
                            for row in primary.read_text().splitlines()))
        self.assertEqual(len((root / "photo_train.txt").read_text().splitlines()), 2)
        self.assertEqual(len((root / "pairs.txt").read_text().splitlines()), 2)
        self.assertEqual(marker.read_text(), "preserve this preview")

    def test_sweep_beta_zero_is_a_true_control_and_metrics_match_member4(self):
        args = parser().parse_args(["sweep", "--data-root", str(self.root), "--limit", "1",
                                   "--content-size", "64", "--style-size", "64", "--threads", "2",
                                   "--betas", "0", "1"])
        summaries = self.invoke(args)
        self.assertEqual([s["beta"] for s in summaries], [0, 1])
        manifest = Path(summaries[0]["manifest"])
        ours = self.root / manifest.read_text().split()[0]
        adain = self.root / (manifest.parent / "manifest_adain.txt").read_text().split()[0]
        self.assertEqual(ours.read_bytes(), adain.read_bytes())
        source = self.root / "PACS" / self.rows[0][0]
        actual = StructureMetrics().compare(source, ours)["edge_score"]
        # Import the unmodified member-4 implementation with its script-style import.
        spec = importlib.util.spec_from_file_location("member4_metric_reference", config.CODE_ROOT / "structure_metrics.py")
        reference = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"config": config}):
            spec.loader.exec_module(reference)
        expected = reference.edge_consistency(reference.edge_map(source), reference.edge_map(ours))
        if np.isnan(expected):
            self.assertIsNone(actual)
        else:
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertIsNotNone(summaries[0]["ours"]["edge_score"])


if __name__ == "__main__":
    unittest.main()
