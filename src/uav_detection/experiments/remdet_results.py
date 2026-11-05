"""将MMEngine测试完成后的指标JSON整理为汇报用CSV；不重算指标。"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    'AP': 'coco/bbox_mAP',
    'AP50': 'coco/bbox_mAP_50',
    'AP75': 'coco/bbox_mAP_75',
    'AP_s': 'coco/bbox_mAP_s',
    'AP_m': 'coco/bbox_mAP_m',
    'AP_l': 'coco/bbox_mAP_l',
}


def summarize(run_dir: Path):
    candidates = []
    # MMEngine 0.10.4 LoggerHook的test输出是时间戳目录下的单个JSON对象。
    for path in run_dir.glob('*/*.json'):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and all(key in payload for key in METRICS.values()):
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise ValueError(f'需要恰好一份完成的test指标JSON，找到{len(candidates)}份；'
                         '请指定单次评估的--run-dir，不要混用多次运行目录')
    source, payload = candidates[0]
    values = {name: float(payload[key]) for name, key in METRICS.items()}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError('AP指标异常（如-1表示无有效GT）；检查数据与评估日志后再生成报告')
    csv_path = run_dir / 'metrics.csv'
    json_path = run_dir / 'metrics_summary.json'
    if csv_path.exists() or json_path.exists():
        raise ValueError('汇总文件已存在，保留原结果；需要重跑时使用新的work-dir')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value_0_to_1', 'value_percent'])
        for name, value in values.items():
            writer.writerow([name, value, round(value * 100, 3)])
    json_path.write_text(json.dumps({
        'source_metrics': str(source), 'metrics_0_to_1': values,
        'metrics_percent': {name: round(value * 100, 3) for name, value in values.items()},
        'note': '仅整理本次test输出；不能据此声称完成从头训练或速度复现。',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, value in values.items():
        print(f'{name}: {value:.3f} ({value * 100:.1f})')
    print('汇总：', csv_path)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_dir.expanduser().resolve())


if __name__ == '__main__':
    main()
