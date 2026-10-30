from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from convert_annotations import convert_split
from data_audit import audit_split
from visdrone_utils import CLASS_NAMES, parse_visdrone_row


class VisDroneParserTests(unittest.TestCase):
    def test_parser_accepts_comma_separated_eight_fields(self) -> None:
        self.assertEqual(
            parse_visdrone_row("12,8,32,16,1,4,0,1"),
            (12.0, 8.0, 32.0, 16.0, 1, 4, 0, 1),
        )

    def test_parser_rejects_malformed_or_non_finite_fields(self) -> None:
        for line in ("1,2,3", "1,2,3,4,1,4,0,inf"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_visdrone_row(line)

    def test_class_names_match_visdrone_one_based_categories(self) -> None:
        self.assertEqual(len(CLASS_NAMES), 10)
        self.assertEqual(CLASS_NAMES[0], "pedestrian")
        self.assertEqual(CLASS_NAMES[9], "motor")


class VisDroneFilePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_dir = self.root / "images"
        self.annotation_dir = self.root / "annotations"
        self.label_dir = self.root / "labels"
        self.image_dir.mkdir()
        self.annotation_dir.mkdir()
        Image.new("RGB", (100, 50), color="white").save(self.image_dir / "sample.jpg")
        (self.annotation_dir / "sample.txt").write_text(
            "10,20,20,10,1,4,0,0\n"
            "0,0,30,20,0,0,0,0\n"
            "0,0,0,0,0,11,0,0\n"
            "90,40,20,20,1,10,0,0\n"
            "1,2,4,0,1,2,0,0\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_audit_recognizes_ignored_categories_and_reports_degenerate_valid_box(self) -> None:
        report = audit_split("train", self.image_dir, self.annotation_dir)
        self.assertEqual(report["image_count"], 1)
        self.assertEqual(report["missing_annotation_count"], 0)
        self.assertEqual(report["annotation_row_count"], 5)
        self.assertEqual(report["ignored_or_unknown_object_count"], 2)
        self.assertEqual(report["invalid_box_count"], 1)
        self.assertEqual(report["unknown_class_count"], 0)
        self.assertEqual(report["class_distribution"]["car"], 1)
        self.assertEqual(report["class_distribution"]["motor"], 1)
        self.assertEqual(report["category_distribution_including_ignored"]["11"], 1)

    def test_conversion_normalizes_coordinates_ignores_regions_and_skips_degenerate_box(self) -> None:
        report = convert_split(
            "train",
            self.image_dir,
            self.annotation_dir,
            self.label_dir,
            overwrite=False,
        )
        rows = (self.label_dir / "sample.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].split()[0], "3")
        self.assertEqual(rows[1].split()[0], "9")
        self.assertEqual(
            tuple(map(float, rows[0].split()[1:])),
            (0.2, 0.5, 0.2, 0.2),
        )
        self.assertEqual(
            tuple(map(float, rows[1].split()[1:])),
            (0.95, 0.9, 0.1, 0.2),
        )
        self.assertEqual(report["ignored_region_count"], 2)
        self.assertEqual(report["invalid_box_count"], 1)
        self.assertEqual(report["clipped_box_count"], 1)

    def test_conversion_refuses_to_overwrite_existing_labels_without_flag(self) -> None:
        self.label_dir.mkdir()
        (self.label_dir / "sample.txt").write_text("do not overwrite", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            convert_split(
                "train",
                self.image_dir,
                self.annotation_dir,
                self.label_dir,
                overwrite=False,
            )
        self.assertEqual(
            (self.label_dir / "sample.txt").read_text(encoding="utf-8"),
            "do not overwrite",
        )


if __name__ == "__main__":
    unittest.main()
