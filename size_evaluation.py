"""Project-specific original-image area evaluation on converted YOLO labels."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visdrone_utils import CLASS_NAMES, list_images


AREA_RANGES = {
    "all": (0.0, math.inf),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, math.inf),
}
IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def in_area_range(box: list[float], area_range: tuple[float, float]) -> bool:
    return area_range[0] <= box_area(box) < area_range[1]


def load_validation_records(image_dir: Path, label_dir: Path) -> tuple[list[dict], str]:
    records = []
    fingerprint = hashlib.sha256()
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"验证图片目录为空：{image_dir}")
    if len({path.stem for path in image_paths}) != len(image_paths):
        raise ValueError("验证图片存在重名 stem，无法唯一对应标签")
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        label_bytes = label_path.read_bytes()
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        targets = []
        for line_number, line in enumerate(label_bytes.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                fields = [float(field) for field in line.split()]
                if len(fields) != 5 or not all(math.isfinite(value) for value in fields):
                    raise ValueError("必须有五个有限数值")
                category, center_x, center_y, box_width, box_height = fields
                if not category.is_integer() or not 0 <= category < len(CLASS_NAMES):
                    raise ValueError("类别不在 0..9 范围")
                if not all(0 <= value <= 1 for value in fields[1:]):
                    raise ValueError("坐标必须归一化到 0..1")
                if box_width <= 0 or box_height <= 0:
                    raise ValueError("框宽高必须为正")
                box = [
                    max(0.0, (center_x - box_width / 2) * width),
                    max(0.0, (center_y - box_height / 2) * height),
                    min(float(width), (center_x + box_width / 2) * width),
                    min(float(height), (center_y + box_height / 2) * height),
                ]
                if box_area(box) <= 0:
                    raise ValueError("框不与图像相交")
            except ValueError as error:
                raise ValueError(f"{label_path}:{line_number}: {error}") from error
            targets.append({"class_id": int(category), "box": box})
        fingerprint.update(image_path.name.encode("utf-8") + b"\0")
        fingerprint.update(file_sha256(image_path).encode("ascii"))
        fingerprint.update(hashlib.sha256(label_bytes).digest())
        records.append({"path": str(image_path.resolve()), "targets": targets})
    return records, fingerprint.hexdigest()


def collect_predictions(
    model: Any, records: list[dict], imgsz: int, batch: int, device: str,
    conf: float, nms_iou: float, max_det: int,
) -> list[list[dict]]:
    predictions = []
    for offset in range(0, len(records), batch):
        group = records[offset : offset + batch]
        results = model.predict(
            source=[record["path"] for record in group], imgsz=imgsz,
            batch=batch, device=device, conf=conf, iou=nms_iou, max_det=max_det,
            rect=False, augment=False, agnostic_nms=False, save=False, verbose=False,
        )
        if len(results) != len(group):
            raise ValueError("预测返回数量与输入图片数量不一致")
        for record, result in zip(group, results):
            if Path(result.path).resolve() != Path(record["path"]).resolve():
                raise ValueError("预测图片顺序与真实标签不一致")
            boxes = result.boxes
            predictions.append([
                {"class_id": int(category), "score": float(score), "box": box}
                for box, score, category in zip(
                    boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                )
            ])
        print(f"尺寸评估推理：{min(offset + batch, len(records))}/{len(records)}")
    return predictions


def match_image(
    targets: list[dict], predictions: list[dict], category: int,
    area_range: tuple[float, float], threshold: float, max_det: int,
) -> list[tuple[float, bool]]:
    ordered_predictions = sorted(predictions, key=lambda item: -item["score"])[:max_det]
    candidates = [target for target in targets if target["class_id"] == category]
    used = set()
    outcomes = []
    for prediction in ordered_predictions:
        if prediction["class_id"] != category:
            continue
        best_index = None
        for ignored in (False, True):
            best_iou = threshold
            for index, target in enumerate(candidates):
                if index in used or (not in_area_range(target["box"], area_range)) != ignored:
                    continue
                overlap = box_iou(prediction["box"], target["box"])
                if overlap >= best_iou:
                    best_index, best_iou = index, overlap
            if best_index is not None:
                break
        if best_index is not None:
            used.add(best_index)
            if in_area_range(candidates[best_index]["box"], area_range):
                outcomes.append((prediction["score"], True))
        elif in_area_range(prediction["box"], area_range):
            outcomes.append((prediction["score"], False))
    return outcomes


def precision_recall_metrics(
    outcomes: list[tuple[float, bool]], target_count: int, recall_conf: float,
) -> tuple[float | None, float | None, float | None]:
    if target_count == 0:
        return None, None, None
    if not outcomes:
        return 0.0, 0.0, 0.0
    ordered = sorted(outcomes, key=lambda item: -item[0])
    true_positives = np.cumsum([correct for _, correct in ordered])
    recalls = true_positives / target_count
    precisions = true_positives / np.arange(1, len(ordered) + 1)
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    indices = np.searchsorted(recalls, np.linspace(0, 1, 101), side="left")
    samples = np.zeros(101)
    valid = indices < len(envelope)
    samples[valid] = envelope[indices[valid]]
    fixed_recall = sum(correct for score, correct in ordered if score >= recall_conf) / target_count
    return float(samples.mean()), float(recalls[-1]), float(fixed_recall)


def evaluate_size_groups(
    records: list[dict], predictions: list[list[dict]], max_det: int = 300,
    recall_conf: float = 0.25,
) -> dict:
    if max_det <= 0 or not 0 <= recall_conf <= 1:
        raise ValueError("max_det 必须为正，recall_conf 必须在 0..1")
    if len(records) != len(predictions):
        raise ValueError("预测数量与验证图片数量不一致")
    groups = {}
    for name, area_range in AREA_RANGES.items():
        per_class = {}
        for category, class_name in enumerate(CLASS_NAMES):
            count = sum(
                target["class_id"] == category and in_area_range(target["box"], area_range)
                for record in records for target in record["targets"]
            )
            scores = []
            for threshold in IOU_THRESHOLDS if count else ():
                outcomes = []
                for record, predicted in zip(records, predictions):
                    outcomes.extend(match_image(
                        record["targets"], predicted, category, area_range, threshold, max_det
                    ))
                scores.append(precision_recall_metrics(outcomes, count, recall_conf))
            per_class[class_name] = {
                "gt_count": count,
                "AP50_95": float(np.mean([score[0] for score in scores])) if count else None,
                "AP50": scores[0][0] if count else None,
                "AR50_95": float(np.mean([score[1] for score in scores])) if count else None,
                "recall50_at_conf": scores[0][2] if count else None,
            }
        valid_classes = [result for result in per_class.values() if result["gt_count"]]
        groups[name] = {
            "gt_count": sum(result["gt_count"] for result in per_class.values()),
            "classes_with_gt": len(valid_classes),
            **{
                metric: float(np.mean([result[metric] for result in valid_classes]))
                if valid_classes else None
                for metric in ("AP50_95", "AP50", "AR50_95", "recall50_at_conf")
            },
            "per_class": per_class,
        }
    return groups
