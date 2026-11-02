import unittest

from analyze_failures import analyze_inputs, diagnose_image


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


if __name__ == "__main__":
    unittest.main()
