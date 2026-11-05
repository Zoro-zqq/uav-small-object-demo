from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from uav_detection.datasets.statistics import load_coco, load_visdrone, write_statistics


class DatasetStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "train.json"
        self.data = {
            "images": [{"id": 10, "file_name": "a.png", "width": 100, "height": 50},
                       {"id": 20, "file_name": "b.png", "width": 100, "height": 50}],
            "categories": [{"id": 7, "name": "car"}, {"id": 42, "name": "ship"}],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 42, "bbox": [0, 0, 16, 4], "area": 9999},
                {"id": 2, "image_id": 10, "category_id": 7, "bbox": [95, 0, 10, 10]},
                {"id": 3, "image_id": 10, "category_id": 7, "bbox": [0, 0, 0, 10]},
                {"id": 4, "image_id": 20, "category_id": 7, "bbox": [0, 0, 10, 10], "iscrowd": 1},
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def load(self, image_dir: Path | None = None):
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        return load_coco(self.path, "test-data", "train", image_dir)

    def test_sparse_categories_bbox_area_and_zero_object_images(self) -> None:
        stats = self.load()
        self.assertEqual(stats.scales, [8.0, 10.0])
        self.assertEqual(stats.image_counts, {"10": 2, "20": 0})
        self.assertEqual(stats.class_counts, {"car": 1, "ship": 1})
        self.assertEqual(stats.issues["invalid_bbox_skipped"], 1)
        self.assertEqual(stats.issues["ignored_or_crowd_skipped"], 1)
        self.assertEqual(stats.issues["out_of_bounds_included_without_clipping"], 1)
        self.assertFalse(stats.image_check["performed"])

    def test_dangling_reference_is_not_silently_counted(self) -> None:
        self.data["annotations"][0]["category_id"] = 1
        with self.assertRaisesRegex(ValueError, "image/category"):
            self.load()

    def test_duplicate_images_and_annotations_are_rejected(self) -> None:
        self.data["images"].append(self.data["images"][0].copy())
        with self.assertRaisesRegex(ValueError, "image id"):
            self.load()
        self.data["images"].pop()
        self.data["annotations"].append(self.data["annotations"][0].copy())
        with self.assertRaisesRegex(ValueError, "annotation id"):
            self.load()

    def test_image_check_records_missing_and_wrong_dimensions(self) -> None:
        images = self.root / "images"
        images.mkdir()
        Image.new("RGB", (20, 20)).save(images / "a.png")
        stats = self.load(images)
        self.assertEqual(stats.image_check["failure_count"], 2)
        self.data["images"][0]["file_name"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "file_name"):
            self.load(images)

    def test_outputs_keep_zero_images_and_refuse_existing_directory(self) -> None:
        stats = self.load()
        output = self.root / "out"
        summary = write_statistics(stats, output)
        self.assertEqual(summary["zero_valid_object_image_count"], 1)
        self.assertEqual(summary["instances_per_image_mean"], 1)
        self.assertEqual(sum(row["count"] for row in summary["size_histogram"]), 2)
        self.assertEqual(sum(row["count"] for row in summary["instances_histogram"]), 2)
        for name in ("object_sizes.png", "instances_per_image.png", "class_long_tail.png"):
            with Image.open(output / name) as image:
                image.verify()
        with self.assertRaises(FileExistsError):
            write_statistics(stats, output)

    def test_visdrone_ignores_regions_and_preserves_empty_images(self) -> None:
        images, annotations = self.root / "images", self.root / "annotations"
        images.mkdir()
        annotations.mkdir()
        for stem in ("a", "b"):
            Image.new("RGB", (100, 50)).save(images / f"{stem}.png")
        (annotations / "a.txt").write_text("0,0,16,4,1,4,0,0\n0,0,20,20,0,0,0,0\n", encoding="utf-8")
        (annotations / "b.txt").write_text("", encoding="utf-8")
        config = self.root / "dataset.yaml"
        config.write_text("path: .\ntrain: images\ntrain_annotations: annotations\n", encoding="utf-8")
        stats = load_visdrone(config, "train")
        self.assertEqual(stats.scales, [8.0])
        self.assertEqual(stats.image_counts, {"a": 1, "b": 0})
        self.assertEqual(stats.class_counts["car"], 1)
        with self.assertRaises(ValueError):
            load_visdrone(config, "all")


if __name__ == "__main__":
    unittest.main()
