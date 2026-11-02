import unittest

from analyze_failures import analyze_inputs, choose_cases, diagnose_image, target_size_bin


def box(category, coordinates, score=None):
    result = {"class_id": category, "box": coordinates}
    if score is not None:
        result["score"] = score
    return result


class FailureAnalysisTests(unittest.TestCase):
    def test_correct_match_takes_priority_over_wrong_class(self):
        target = box(3, [0, 0, 10, 10])
        events = diagnose_image([target], [box(4, target["box"], 0.99), box(3, target["box"], 0.8)])
        self.assertEqual([event["kind"] for event in events], ["correct", "unmatched_prediction"])

    def test_class_confusion_localization_and_unmatched_are_exclusive(self):
        targets = [box(3, [0, 0, 10, 10]), box(0, [30, 0, 40, 10]), box(1, [60, 0, 70, 10])]
        predicted = [box(4, targets[0]["box"], 0.9), box(0, [30, 0, 34, 10], 0.8)]
        events = diagnose_image(targets, predicted)
        self.assertEqual([event["kind"] for event in events], ["class_confusion", "localization", "miss"])
        self.assertEqual(sum(event["target"] is not None for event in events), len(targets))
        self.assertEqual(sum(event["prediction"] is not None for event in events), len(predicted))

    def test_low_confidence_is_not_displayed_and_small_groups_use_gt(self):
        data = {"records": [{"path": "/data/a.jpg", "targets": [box(0, [0, 0, 10, 10])]}],
                "predictions": [[box(0, [0, 0, 10, 10], 0.2)]]}
        result = analyze_inputs(data, 0.25)
        self.assertEqual(result["counts"]["miss"], 1)
        self.assertEqual(result["counts"]["unmatched_prediction"], 0)
        self.assertEqual(result["small_gt_counts"]["miss"], 1)

    def test_background_image_does_not_crash(self):
        self.assertEqual(diagnose_image([], []), [])
        events = diagnose_image([], [box(3, [0, 0, 10, 10], 0.9)])
        self.assertEqual(events[0]["kind"], "unmatched_prediction")

    def test_size_boundaries_and_micro_recall(self):
        self.assertEqual(target_size_bin([0, 0, 8, 8]), "8-16")
        self.assertEqual(target_size_bin([0, 0, 32, 32]), "32-96")
        self.assertEqual(target_size_bin([0, 0, 96, 96]), "96+")
        data = {"records": [{"path": "a.jpg", "targets": [box(0, [0, 0, 10, 10]), box(0, [30, 0, 40, 10])]}],
                "predictions": [[box(0, [0, 0, 10, 10], 0.9)]]}
        bins = analyze_inputs(data, 0.25)["size_bins"]
        self.assertEqual(bins["8-16"]["micro_recall50_at_conf"], 0.5)
        self.assertIsNone(bins["0-8"]["micro_recall50_at_conf"])

    def test_twenty_cases_cover_classes_and_are_deterministic(self):
        events = []
        for kind in ("miss", "class_confusion", "localization", "unmatched_prediction"):
            for index in range(20):
                target = box(index % 10, [0, 0, 10 + index * 5, 10 + index * 5])
                predicted = {**target, "score": 0.9 - index / 100}
                events.append({"kind": kind, "image": f"{kind}_{index}.jpg",
                               "target": None if kind == "unmatched_prediction" else target,
                               "prediction": None if kind == "miss" else predicted})
        cases = choose_cases(events, 20)
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({event["image"] for event in cases}), 20)
        self.assertEqual(cases, choose_cases(events, 20))
        self.assertEqual(len({event["target"]["class_id"] for event in cases if event["kind"] == "miss"}), 6)
        self.assertEqual(len(choose_cases(events)), 10)

    def test_case_selection_does_not_duplicate_images_to_fill_quota(self):
        events = [{"kind": "miss", "image": "same.jpg", "target": box(0, [0, 0, 10, 10]), "prediction": None}] * 30
        self.assertEqual(len(choose_cases(events, 20)), 1)
        with self.assertRaises(ValueError):
            choose_cases(events, 15)


if __name__ == "__main__":
    unittest.main()
