"""将原生 VisDrone DET train/val 转为可审计的 COCO bbox 标注。"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image

from uav_detection.datasets.visdrone import CLASS_NAMES


CLASSES = tuple(CLASS_NAMES)
PROTOCOL = 'valid10_drop_ignored_v1'


def find_split(root, split, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / 'images').is_dir() or not (path / 'annotations').is_dir():
            raise ValueError(f'{path}: 需要同级 images/ 和 annotations/')
        return path
    names = {split, f'VisDrone2019-DET-{split}'}
    candidates = []
    parents = [root]
    parents.extend(p for p in root.iterdir() if p.is_dir())
    for parent in parents:
        if parent.name in names:
            candidates.append(parent)
        candidates.extend(parent / name for name in names)
    found = sorted({p.resolve() for p in candidates
                    if (p / 'images').is_dir() and (p / 'annotations').is_dir()})
    if len(found) != 1:
        raise ValueError(
            f'{split}: 找到 {len(found)} 个标准目录: {found}。'
            f'请用 --{split}-dir 显式指定含 images/、annotations/ 的目录。')
    return found[0]


def parse_row(line, location):
    fields = line.rstrip(',').split(',')
    if len(fields) != 8:
        raise ValueError(f'{location}: 原生标注应有8列，实际{len(fields)}列')
    try:
        values = [float(value.strip()) for value in fields]
    except ValueError as exc:
        raise ValueError(f'{location}: 非数值标注') from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{location}: 存在非有限数值')
    for index in (4, 5, 6, 7):
        if not values[index].is_integer():
            raise ValueError(f'{location}: score/category/truncation/occlusion必须为整数')
    x, y, w, h = values[:4]
    score, category, truncation, occlusion = map(int, values[4:])
    if score not in (0, 1) or not 0 <= category <= 11:
        raise ValueError(f'{location}: 非预期score或category；确认输入为train/val真值标注')
    return x, y, w, h, score, category, truncation, occlusion


def convert_split(source, split):
    image_dir = source / 'images'
    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    if not images:
        raise ValueError(f'{image_dir}: 没有找到图片')
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError(f'{image_dir}: 存在重复stem，无法唯一关联TXT')
    annotation_stems = {p.stem for p in (source / 'annotations').glob('*.txt')}
    missing = set(stems) - annotation_stems
    extra = annotation_stems - set(stems)
    if missing or extra:
        raise ValueError(f'{split}: 图片/TXT不一一对应；缺TXT={sorted(missing)[:5]}，'
                         f'无对应图片的TXT={sorted(extra)[:5]}')
    coco = {
        'info': {'description': f'VisDrone2019-DET {split}',
                 'conversion_protocol': PROTOCOL},
        'licenses': [], 'images': [], 'annotations': [],
        'categories': [{'id': i, 'name': name, 'supercategory': 'object'}
                       for i, name in enumerate(CLASSES, 1)],
    }
    stats = Counter(images=len(images))
    per_class = Counter()
    excluded = []
    annotation_id = 1
    for image_id, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco['images'].append({'id': image_id, 'file_name': image_path.name,
                               'width': width, 'height': height})
        txt = source / 'annotations' / (image_path.stem + '.txt')
        kept_in_image = 0
        for line_no, line in enumerate(txt.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            row = parse_row(line, f'{txt}:{line_no}')
            x, y, w, h, score, category, truncation, occlusion = row
            stats['source_rows'] += 1
            reason = None
            if category == 0:
                reason = 'ignored_region_category0'
            elif category == 11:
                reason = 'others_category11'
            elif score == 0:
                reason = 'score0_ignored'
            elif w < 1 or h < 1:
                reason = 'invalid_bbox_size'
            elif min(x + w, width) <= max(x, 0) or min(y + h, height) <= max(y, 0):
                reason = 'bbox_outside_image'
            if reason:
                stats[reason] += 1
                excluded.append({'file_name': image_path.name, 'line': line_no,
                                 'reason': reason, 'source_values': list(row)})
                continue
            # 保留原生xywh，不擅自减1或裁剪边界；训练框架会再次检查相交与尺寸。
            if x < 0 or y < 0 or x + w > width or y + h > height:
                stats['kept_bbox_crossing_image_boundary'] += 1
            coco['annotations'].append({
                'id': annotation_id, 'image_id': image_id,
                'category_id': category, 'bbox': [x, y, w, h],
                'area': w * h, 'iscrowd': 0,
                'visdrone_truncation': truncation, 'visdrone_occlusion': occlusion,
            })
            annotation_id += 1
            kept_in_image += 1
            per_class[CLASSES[category - 1]] += 1
        if kept_in_image == 0:
            stats['images_without_valid_boxes'] += 1
    stats['kept_boxes'] = len(coco['annotations'])
    return coco, {'source': str(source), 'counts': dict(stats),
                  'per_class': {name: per_class[name] for name in CLASSES}}, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['all', 'train', 'val'], default='all',
                        help='默认转换train和val；仅评估权重时可只转换val')
    parser.add_argument('--train-dir')
    parser.add_argument('--val-dir')
    parser.add_argument('--ignore-policy', choices=['drop'], required=True,
                        help='显式采用有效10类COCO转换；不等价于原生忽略区域评价')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise ValueError(f'拒绝覆盖已有输出目录：{out}；请选择新的输出路径')
    if not root.is_dir():
        raise ValueError(f'输入根目录不存在：{root}')
    selected = ('train', 'val') if args.split == 'all' else (args.split,)
    splits = {name: find_split(root, name, getattr(args, name + '_dir'))
              for name in selected}
    if len(splits) == 2 and splits['train'] == splits['val']:
        raise ValueError('train和val不能指向同一目录')
    # 先解析完所选输入，再创建输出；错误标注不会产生可误用的半份JSON。
    converted = {name: convert_split(path, name) for name, path in splits.items()}
    (out / 'images').mkdir(parents=True)
    (out / 'annotations').mkdir()
    report = {'protocol': PROTOCOL,
              'warning': '原论文未提供已核实的转换脚本；本协议排除忽略标注，'
                         '不复刻原生VisDrone忽略区域规则；绝对AP可能不与作者JSON一致。',
              'splits': {}}
    for name, (coco, stats, excluded) in converted.items():
        (out / 'images' / name).symlink_to(splits[name] / 'images', target_is_directory=True)
        path = out / 'annotations' / f'VisDrone2019-DET_{name}_coco.json'
        path.write_text(json.dumps(coco, ensure_ascii=False), encoding='utf-8')
        (out / 'annotations' / f'{name}_excluded_rows.json').write_text(
            json.dumps(excluded, ensure_ascii=False), encoding='utf-8')
        report['splits'][name] = stats
        print(name, json.dumps(stats, ensure_ascii=False))
    (out / 'conversion_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('输出：', out)
    print(report['warning'])


if __name__ == '__main__':
    main()
