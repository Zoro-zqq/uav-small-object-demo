from __future__ import annotations

import tempfile
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import evaluate
from size_evaluation import collect_predictions, evaluate_size_groups, load_validation_records


def target(box: list[float], category: int = 3) -> dict:
    return {"class_id": category, "box": box}


def prediction(box: list[float], score: float = 0.9, category: int = 3) -> dict:
    return {**target(box, category), "score": score}


class SizeEvaluationTests(unittest.TestCase):
    def test_perfect_predictions_for_each_exclusive_area_bucket(self) -> None:
        boxes = [[0, 0, 10, 10], [100, 0, 132, 32], [200, 0, 296, 96]]
        groups = evaluate_size_groups(
            [{"targets": [target(box) for box in boxes]}],
            [[prediction(box) for box in boxes]],
        )
        for group in ("small", "medium", "large"):
            self.assertEqual(groups[group]["gt_count"], 1)
            self.assertAlmostEqual(groups[group]["AP50_95"], 1.0)
        self.assertEqual(groups["all"]["gt_count"], 3)

    def test_out_of_range_matches_do_not_become_small_false_positives(self) -> None:
        small = [0, 0, 10, 10]
        large = [100, 100, 300, 300]
        groups = evaluate_size_groups(
            [{"targets": [target(small), target(large)]}],
            [[prediction(large, 0.99), prediction(small, 0.8)]],
        )
        self.assertAlmostEqual(groups["small"]["AP50"], 1.0)

    def test_small_false_positive_before_true_positive_reduces_ap(self) -> None:
        box = [0, 0, 10, 10]
        groups = evaluate_size_groups(
            [{"targets": [target(box)]}],
            [[prediction([30, 30, 40, 40], 0.99), prediction(box, 0.8)]],
        )
        self.assertAlmostEqual(groups["small"]["AP50"], 0.5)

    def test_empty_predictions_and_absent_groups_are_not_confused(self) -> None:
        groups = evaluate_size_groups([{"targets": [target([0, 0, 10, 10])]}], [[]])
        self.assertEqual(groups["small"]["AP50"], 0.0)
        self.assertEqual(groups["small"]["recall50_at_conf"], 0.0)
        self.assertIsNone(groups["large"]["AP50"])

    def test_duplicate_predictions_cannot_match_one_target_twice(self) -> None:
        box = [0, 0, 10, 10]
        groups = evaluate_size_groups(
            [{"targets": [target(box), target([50, 50, 60, 60])]}],
            [[prediction(box, 0.9), prediction(box, 0.8)]],
        )
        self.assertEqual(groups["small"]["recall50_at_conf"], 0.5)

    def test_wrong_category_is_not_a_true_positive(self) -> None:
        box = [0, 0, 10, 10]
        groups = evaluate_size_groups(
            [{"targets": [target(box)]}], [[prediction(box, category=4)]]
        )
        self.assertEqual(groups["small"]["AP50"], 0.0)

    def test_confidence_recall_and_detection_limit_are_explicit(self) -> None:
        box = [0, 0, 10, 10]
        records = [{"targets": [target(box)]}]
        predicted = [[prediction([30, 30, 40, 40], 0.9), prediction(box, 0.2)]]
        unlimited = evaluate_size_groups(records, predicted)
        limited = evaluate_size_groups(records, predicted, max_det=1)
        self.assertEqual(unlimited["small"]["AR50_95"], 1.0)
        self.assertEqual(unlimited["small"]["recall50_at_conf"], 0.0)
        self.assertEqual(limited["small"]["AR50_95"], 0.0)

    def test_background_images_contribute_false_positives(self) -> None:
        box = [0, 0, 10, 10]
        groups = evaluate_size_groups(
            [{"targets": []}, {"targets": [target(box)]}],
            [[prediction(box, 0.99)], [prediction(box, 0.8)]],
        )
        self.assertAlmostEqual(groups["small"]["AP50"], 0.5)

    def test_stricter_iou_penalizes_localization_and_averages_only_present_classes(self) -> None:
        groups = evaluate_size_groups(
            [{"targets": [target([0, 0, 10, 10])]}],
            [[prediction([0, 0, 7, 10])]],
        )
        self.assertEqual(groups["small"]["classes_with_gt"], 1)
        self.assertEqual(groups["small"]["AP50"], 1.0)
        self.assertAlmostEqual(groups["small"]["AP50_95"], 0.5)

    def test_prediction_adapter_batches_and_pins_settings(self) -> None:
        calls = []

        class Values:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Model:
            def predict(self, **kwargs):
                calls.append(kwargs)
                return [SimpleNamespace(
                    path=path, boxes=SimpleNamespace(
                        xyxy=Values([[0, 0, 10, 10]]),
                        conf=Values([0.9]), cls=Values([3]),
                    ),
                ) for path in kwargs["source"]]

        records = [{"path": str(Path(f"sample_{index}.jpg").resolve())} for index in range(3)]
        results = collect_predictions(Model(), records, 960, 2, "cpu", 0.001, 0.7, 300)
        self.assertEqual(len(results), 3)
        self.assertEqual([len(call["source"]) for call in calls], [2, 1])
        self.assertFalse(calls[0]["rect"])
        self.assertEqual(results[0][0]["class_id"], 3)

    def test_loader_restores_original_pixels_and_fingerprints_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (2000, 1000)).save(images / "sample.jpg")
            label = labels / "sample.txt"
            label.write_text("3 0.5 0.5 0.01 0.01\n", encoding="utf-8")
            records, fingerprint = load_validation_records(images, labels)
            self.assertEqual(records[0]["targets"][0]["box"], [990, 495, 1010, 505])
            label.write_text("", encoding="utf-8")
            empty_records, changed = load_validation_records(images, labels)
            self.assertNotEqual(fingerprint, changed)
            self.assertEqual(empty_records[0]["targets"], [])
            label.unlink()
            with self.assertRaises(FileNotFoundError):
                load_validation_records(images, labels)

    def test_cli_writes_size_reports_with_mocked_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (100, 50)).save(images / "sample.jpg")
            (labels / "sample.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            config = root / "dataset.yaml"
            config.write_text(f"path: {root.as_posix()}\nval: images\n", encoding="utf-8")
            weights = root / "best.pt"
            weights.write_bytes(b"mock checkpoint")
            output = root / "results"

            class Model:
                def __init__(self, path):
                    self.path = path

                def val(self, **kwargs):
                    return SimpleNamespace(box=SimpleNamespace(mp=0.5, mr=0.5, map50=0.5, map=0.3))

            argv = [
                "evaluate.py", "--weights", str(weights), "--data", str(config),
                "--output-dir", str(output), "--size-groups", "--device", "cpu",
            ]
            with patch.object(sys, "argv", argv), patch.dict(
                sys.modules, {"ultralytics": SimpleNamespace(YOLO=Model)}
            ), patch("evaluate.version", return_value="mock"), patch(
                "size_evaluation.collect_predictions",
                return_value=[[prediction([40, 20, 60, 30])]],
            ):
                evaluate.main()
            report = json.loads((output / "best_size_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["groups"]["small"]["AP50"], 1.0)
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(len(report["validation_images_and_labels_sha256"]), 64)
            self.assertTrue((output / "best_size_metrics.csv").is_file())
            self.assertTrue((output / "best_size_inputs.json").is_file())
            self.assertTrue((output / "best_val_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
