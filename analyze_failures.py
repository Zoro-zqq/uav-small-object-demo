"""Diagnose saved predictions without training or GPU inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from size_evaluation import box_area, load_validation_records
from visdrone_utils import CLASS_NAMES


ERROR_NAMES = {
    "correct": "正确匹配",
    "class_confusion": "错类别候选",
    "localization": "定位偏差候选",
    "miss": "未匹配真实框",
    "unmatched_prediction": "未匹配预测框",
}


def diagnose_image(targets: list[dict], predictions: list[dict], conf: float = 0.25) -> list[dict]:
    predicted = sorted(
        [item for item in predictions if item["score"] >= conf],
        key=lambda item: -item["score"],
    )
    overlaps = np.zeros((len(predicted), len(targets)))
    if predicted and targets:
        predicted_boxes = np.array([item["box"] for item in predicted])
        target_boxes = np.array([item["box"] for item in targets])
        intersections = np.maximum(
            0, np.minimum(predicted_boxes[:, None, 2:], target_boxes[None, :, 2:])
            - np.maximum(predicted_boxes[:, None, :2], target_boxes[None, :, :2]),
        ).prod(axis=2)
        unions = (
            np.array([box_area(item["box"]) for item in predicted])[:, None]
            + np.array([box_area(item["box"]) for item in targets])[None, :] - intersections
        )
        overlaps = np.divide(intersections, unions, out=overlaps, where=unions > 0)
    used_targets, used_predictions = set(), set()
    events = []
    for kind in ("correct", "class_confusion", "localization"):
        for predicted_index, prediction in enumerate(predicted):
            if predicted_index in used_predictions:
                continue
            candidates = []
            for target_index, target in enumerate(targets):
                if target_index in used_targets:
                    continue
                same_class = target["class_id"] == prediction["class_id"]
                overlap = float(overlaps[predicted_index, target_index])
                eligible = (
                    (kind == "correct" and same_class and overlap >= 0.5)
                    or (kind == "class_confusion" and not same_class and overlap >= 0.5)
                    or (kind == "localization" and same_class and 0.1 <= overlap < 0.5)
                )
                if eligible:
                    candidates.append((overlap, target_index))
            if candidates:
                overlap, target_index = max(candidates)
                used_targets.add(target_index)
                used_predictions.add(predicted_index)
                events.append({"kind": kind, "target": targets[target_index], "prediction": prediction, "iou": overlap})
    events.extend(
        {"kind": "miss", "target": target, "prediction": None, "iou": None}
        for index, target in enumerate(targets) if index not in used_targets
    )
    events.extend(
        {"kind": "unmatched_prediction", "target": None, "prediction": prediction, "iou": None}
        for index, prediction in enumerate(predicted) if index not in used_predictions
    )
    return events


def analyze_inputs(data: dict, conf: float) -> dict:
    if len(data["records"]) != len(data["predictions"]):
        raise ValueError("真实框和预测图片数量不一致")
    counts = Counter({kind: 0 for kind in ERROR_NAMES})
    small_counts = Counter({kind: 0 for kind in ERROR_NAMES if kind != "unmatched_prediction"})
    confusions = Counter()
    events = []
    for record, predicted in zip(data["records"], data["predictions"]):
        for event in diagnose_image(record["targets"], predicted, conf):
            event["image"] = record["path"].replace("\\", "/").rsplit("/", 1)[-1]
            counts[event["kind"]] += 1
            target = event["target"]
            if target is not None and box_area(target["box"]) < 1024:
                small_counts[event["kind"]] += 1
            if event["kind"] == "class_confusion":
                confusions[(CLASS_NAMES[target["class_id"]], CLASS_NAMES[event["prediction"]["class_id"]])] += 1
            if event["kind"] != "correct":
                events.append(event)
    return {
        "image_count": len(data["records"]), "counts": dict(counts),
        "small_gt_counts": dict(small_counts),
        "confusions": [{"true": pair[0], "predicted": pair[1], "count": count} for pair, count in confusions.most_common()],
        "events": events,
    }


def choose_cases(events: list[dict]) -> list[dict]:
    selected, used_images = [], set()
    for kind, limit in (("miss", 3), ("class_confusion", 3), ("localization", 2), ("unmatched_prediction", 2)):
        candidates = [event for event in events if event["kind"] == kind]
        if kind == "miss":
            candidates.sort(key=lambda event: (
                box_area(event["target"]["box"]) >= 1024,
                -box_area(event["target"]["box"]), event["image"],
            ))
        else:
            candidates.sort(key=lambda event: (-event["prediction"]["score"], event["image"]))
        taken = 0
        for event in candidates:
            if event["image"] in used_images:
                continue
            selected.append(event)
            used_images.add(event["image"])
            taken += 1
            if taken == limit:
                break
    return selected


def render_case(event: dict, record: dict, predictions: list[dict], image_dir: Path, output: Path, conf: float) -> None:
    focus = event["target"] or event["prediction"]
    with Image.open(image_dir / event["image"]) as source:
        source.load()
        image = source.convert("RGB")
    boxes = [item["box"] for item in (event["target"], event["prediction"]) if item]
    left = max(0, int(min(box[0] for box in boxes)) - 60)
    top = max(0, int(min(box[1] for box in boxes)) - 60)
    right = min(image.width, int(max(box[2] for box in boxes)) + 61)
    bottom = min(image.height, int(max(box[3] for box in boxes)) + 61)
    crop = image.crop((left, top, right, bottom))
    scale = min(3.0, 650 / max(crop.size))
    pane_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
    panes = []
    for title, items, color, chosen in (
        ("GT (converted labels)", record["targets"], "lime", event["target"]),
        (f"Prediction score >= {conf}", [item for item in predictions if item["score"] >= conf], "red", event["prediction"]),
    ):
        pane = Image.new("RGB", (pane_size[0], pane_size[1] + 30), "black")
        pane.paste(crop.resize(pane_size), (0, 30))
        draw = ImageDraw.Draw(pane)
        draw.text((5, 5), title, fill="white")
        for item in items:
            box = item["box"]
            if box[2] < left or box[0] > right or box[3] < top or box[1] > bottom:
                continue
            rectangle = [(box[0] - left) * scale, (box[1] - top) * scale + 30,
                         (box[2] - left) * scale, (box[3] - top) * scale + 30]
            draw.rectangle(rectangle, outline=color, width=4 if item == chosen else 1)
            if item == chosen:
                label = CLASS_NAMES[item["class_id"]]
                if "score" in item:
                    label += f" {item['score']:.3f}"
                draw.text((max(0, rectangle[0]), max(30, rectangle[1] - 12)), label, fill="white", stroke_width=1, stroke_fill="black")
        panes.append(pane)
    canvas = Image.new("RGB", (pane_size[0] * 2, pane_size[1] + 60), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"{event['kind']} | {event['image']} | focus class: {CLASS_NAMES[focus['class_id']]}", fill="white")
    for index, pane in enumerate(panes):
        canvas.paste(pane, (index * pane_size[0], 30))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="从尺寸评估预测存档生成错误统计与候选对照图，无需 GPU")
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/failures/baseline_v1"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("conf 必须在 0..1")
    folders = ("yolov8n_30ep_size_v1", "yolo11n_30ep_size_v1")
    summaries, data_sets, reports = {}, {}, []
    for folder in folders:
        path = args.metrics_root / folder
        reports.append(json.loads((path / "best_size_summary.json").read_text(encoding="utf-8")))
        data_sets[folder] = json.loads((path / "best_size_inputs.json").read_text(encoding="utf-8"))
        summaries[folder] = analyze_inputs(data_sets[folder], args.conf)
    for key in ("protocol", "environment", "validation_images_and_labels_sha256"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"两个模型的 {key} 不一致，停止比较")
    if data_sets[folders[0]]["records"] != data_sets[folders[1]]["records"]:
        raise ValueError("两个存档的真实框或图片顺序不一致")
    if args.conf < max(report["settings"]["conf"] for report in reports):
        raise ValueError("诊断阈值低于存档采集阈值，缺少预测数据")
    visual_provenance = {"content_fingerprint_matches": None, "target_coordinates_match": None}
    if args.image_dir and (args.image_dir.parent / "labels").is_dir():
        local_records, fingerprint = load_validation_records(args.image_dir, args.image_dir.parent / "labels")
        visual_provenance = {
            "content_fingerprint_matches": fingerprint == reports[0]["validation_images_and_labels_sha256"],
            "target_coordinates_match": (
                len(local_records) == len(data_sets[folders[1]]["records"])
                and all(
                    Path(local["path"]).name == saved["path"].replace("\\", "/").rsplit("/", 1)[-1]
                    and local["targets"] == saved["targets"]
                    for local, saved in zip(local_records, data_sets[folders[1]]["records"])
                )
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = data_sets[folders[1]]
    records = {record["path"].replace("\\", "/").rsplit("/", 1)[-1]: (record, predicted)
               for record, predicted in zip(latest["records"], latest["predictions"])}
    cases = choose_cases(summaries[folders[1]]["events"])
    case_rows = []
    for index, event in enumerate(cases, 1):
        target, predicted = event["target"], event["prediction"]
        row = {
            "case": index, "image": event["image"], "error": ERROR_NAMES[event["kind"]],
            "true_class": CLASS_NAMES[target["class_id"]] if target else "",
            "predicted_class": CLASS_NAMES[predicted["class_id"]] if predicted else "",
            "score": predicted["score"] if predicted else "", "iou": event["iou"],
            "gt_box": json.dumps(target["box"]) if target else "",
            "gt_area": box_area(target["box"]) if target else "",
            "review": "待人工复核；不推断遮挡、模糊、漏标等原因", "card": "",
        }
        if args.image_dir:
            image_path = args.image_dir / event["image"]
            if image_path.is_file():
                output = args.output_dir / f"case_{index:02d}.png"
                render_case(event, *records[event["image"]], args.image_dir, output, args.conf)
                row["card"] = output.name
        case_rows.append(row)
    with (args.output_dir / "cases.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(case_rows[0]) if case_rows else ["case"])
        writer.writeheader()
        writer.writerows(case_rows)
    (args.output_dir / "diagnostics.json").write_text(json.dumps({
        "conf": args.conf, "matching_iou": 0.5, "localization_iou": "[0.1,0.5)",
        "validation_fingerprint": reports[0]["validation_images_and_labels_sha256"],
        "local_visual_provenance": visual_provenance,
        "models": {name: {key: value for key, value in result.items() if key != "events"} for name, result in summaries.items()},
        "cases": case_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 失败案例自动诊断（候选，不代替官方指标）", "",
        f"使用完整 {summaries[folders[1]]['image_count']} 张验证图的预测存档，展示阈值 score>={args.conf}。先匹配同类 IoU>=0.5，再配异类 IoU>=0.5，最后配同类 0.1<=IoU<0.5；各框最多配一次。",
        "错类、定位候选单独计数，剩余框才计未匹配，因此下表不能直接当作常规 FN/FP。高密度重叠框的配对存在歧义。",
        "原始忽略区域未进入真实标签，未匹配预测可能是忽略目标或漏标，不能直接全部归因为背景误检。低于阈值的预测视为未显示；未匹配真实框不代表模型在任何阈值下都找不到它。", "",
        f"本地可视化数据检查：{visual_provenance}。若内容指纹未确认一致，局部图只作初步复核，最终在训练服务器相同数据上重生成；汇总数字始终来自服务器保存的真实框与预测，不受本地图片影响。", "",
        "| 诊断类型 | YOLOv8n 30ep | YOLO11n 30ep |", "|---|---:|---:|",
    ]
    for kind, label in ERROR_NAMES.items():
        lines.append(f"| {label} | {summaries[folders[0]]['counts'][kind]} | {summaries[folders[1]]['counts'][kind]} |")
    lines += ["", "## 小目标真实框的诊断分布", "", "| 类型 | YOLOv8n | YOLO11n |", "|---|---:|---:|"]
    for kind in summaries[folders[0]]["small_gt_counts"]:
        lines.append(f"| {ERROR_NAMES[kind]} | {summaries[folders[0]]['small_gt_counts'][kind]} | {summaries[folders[1]]['small_gt_counts'][kind]} |")
    lines += ["", "## YOLO11 错类别候选前十项", "", "| 真实类别 | 预测类别 | 框数 |", "|---|---|---:|"]
    for item in summaries[folders[1]]["confusions"][:10]:
        lines.append(f"| {item['true']} | {item['predicted']} | {item['count']} |")
    lines += ["", "## 10 个候选对照案例", "", "固定选择：3 个未匹配真实框（优先面积较大的小目标）、3 个错类、2 个定位偏差、2 个未匹配预测（后三类按置信度降序）；每图最多选一次。不是随机抽样，不能从这十图估计错误率。", "", "| 编号 | 图片 | 类型 | 真实→预测 | 对照图 |", "|---|---|---|---|---|"]
    for row in case_rows:
        card = f"[打开]({row['card']})" if row["card"] else "图片未找到，需服务器生成"
        lines.append(f"| {row['case']} | {row['image']} | {row['error']} | {row['true_class']} → {row['predicted_class']} | {card} |")
    lines += ["", "## 可以与不可以下的结论", "", "可报告固定阈值下的错误数量、类别混淆方向及目标尺寸；不能仅由框匹配推断是缩放、遮挡、样本不均衡或标注错误导致，也不能将类别汇总差异解释为因果。", "人工只需优先复核这十张局部对照卡：左侧真实框、右侧预测框，重点框线加粗；裁剪和放大仅用于展示，不代表模型实际输入。", "先在 cases.csv 的 review 栏记录能观察到的现象，再写待验证假设。"]
    report_path = args.output_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"诊断报告：{report_path}；候选案例：{len(case_rows)}；对照图：{sum(bool(row['card']) for row in case_rows)}")


if __name__ == "__main__":
    main()
