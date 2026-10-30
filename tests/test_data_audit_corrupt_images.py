from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_audit import audit_split, visualize_annotations


class CorruptImageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_dir = self.root / "images"
        self.annotation_dir = self.root / "annotations"
        self.image_dir.mkdir()
        self.annotation_dir.mkdir()
        Image.new("RGB", (100, 50), color="white").save(self.image_dir / "valid.jpg")
        self.truncated_path = self.image_dir / "truncated.jpg"
        self.truncated_path.write_bytes((self.image_dir / "valid.jpg").read_bytes()[:-2])
        for stem in ("valid", "truncated"):
            (self.annotation_dir / f"{stem}.txt").write_text(
                "10,10,20,10,1,4,0,0\n", encoding="utf-8"
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_audit_detects_truncation_that_verify_misses(self) -> None:
        with Image.open(self.truncated_path) as image:
            image.verify()
        report = audit_split("train", self.image_dir, self.annotation_dir)
        self.assertEqual(report["corrupt_image_count"], 1)
        self.assertIn(str(self.truncated_path), report["corrupt_image_examples"][0])
        self.assertEqual(report["image_size_distribution"], {"100x50": 1})

    def test_visualization_marks_bad_image_and_keeps_source_unchanged(self) -> None:
        original_bytes = self.truncated_path.read_bytes()
        output_path = self.root / "sample.png"
        result = visualize_annotations(
            "train", self.image_dir, self.annotation_dir, output_path, 2, 42
        )
        self.assertEqual(result, output_path)
        self.assertTrue(output_path.is_file())
        with Image.open(output_path) as image:
            image.load()
        self.assertEqual(self.truncated_path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
