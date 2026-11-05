"""为官方RemDet权重生成仅含验证集的评估配置，不启动训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_detection.datasets.converters.visdrone_to_coco import CLASSES, PROTOCOL


SOURCES = {
    'tiny': 'remdet_tiny-300e_coco.py',
    's': 'remdet_s-300e_visdrone.py',
    'm': 'remdet_m-300e_coco.py',
    'l': 'remdet_l-300e_coco.py',
    'x': 'remdet_x-300e_coco.py',
}


def prepare(repo: Path, data_root: Path, output: Path, model: str, workers: int):
    from mmengine.config import Config

    if output.exists():
        raise ValueError(f'拒绝覆盖输出目录：{output}；请换一个--out')
    report = json.loads((data_root / 'conversion_report.json').read_text(encoding='utf-8'))
    if report.get('protocol') != PROTOCOL or 'val' not in report.get('splits', {}):
        raise ValueError('需要配套convert-coco生成的val及转换报告')
    annotation = data_root / 'annotations/VisDrone2019-DET_val_coco.json'
    coco = json.loads(annotation.read_text(encoding='utf-8'))
    categories = {item['id']: item['name'] for item in coco['categories']}
    if categories != dict(enumerate(CLASSES, 1)):
        raise ValueError('验证集不是预期的VisDrone十类及category id')
    image_root = data_root / 'images/val'
    if not coco['images'] or any(not (image_root / item['file_name']).is_file()
                                 for item in coco['images']):
        raise ValueError('验证集图片为空或JSON指向的图片不存在；检查软链接')

    source = repo / 'config_remdet/remdet' / SOURCES[model]
    cfg = Config.fromfile(str(source))
    if (cfg.model.backbone.type != 'RemNet' or
            cfg.model.neck.type != 'RemDetPAFPN' or
            cfg.model.bbox_head.head_module.num_classes != len(CLASSES)):
        raise ValueError(f'来源配置不是完整VisDrone RemDet：{source}')
    loader = cfg.test_dataloader
    loader.batch_size = 1
    loader.num_workers = workers
    loader.persistent_workers = workers > 0
    loader.dataset.data_root = str(data_root) + '/'
    loader.dataset.ann_file = 'annotations/VisDrone2019-DET_val_coco.json'
    loader.dataset.data_prefix.img = 'images/val'
    loader.dataset.batch_shapes_cfg = None
    loader.dataset.pop('indices', None)
    cfg.test_evaluator.ann_file = str(annotation)
    cfg.test_evaluator.classwise = True
    cfg.test_evaluator.outfile_prefix = str(output / 'detections')
    cfg.data_root = str(data_root) + '/'
    # 纯test入口无需训练集、优化器及验证循环；保留来源的模型、测试pipeline、NMS和评价参数。
    for key in ('train_dataloader', 'train_cfg', 'optim_wrapper', 'param_scheduler',
                'val_dataloader', 'val_cfg', 'val_evaluator'):
        cfg[key] = None
    cfg.resume = False
    cfg.load_from = None
    cfg.work_dir = str(output)
    output.mkdir(parents=True)
    config_path = output / 'eval_config.py'
    cfg.dump(str(config_path))
    manifest = {
        'purpose': 'official checkpoint evaluation only; no training',
        'model': model,
        'source_config': str(source),
        'conversion_protocol': report['protocol'],
        'conversion_report': str(data_root / 'conversion_report.json'),
        'validation_images': len(coco['images']),
        'batch_size': 1,
        'test_pipeline': cfg.test_pipeline,
        'model_test_cfg': dict(cfg.model.test_cfg),
        'proposal_nums': list(cfg.test_evaluator.proposal_nums),
        'note': '作者转换JSON尚未核实，本转换协议的AP不保证与论文完全一致；不启用TTA。',
    }
    (output / 'eval_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('评估配置：', config_path)
    print('验证集图片：', len(coco['images']))
    print('模型宽/深：', cfg.model.backbone.widen_factor, cfg.model.backbone.deepen_factor)
    print('保留proposal_nums：', cfg.test_evaluator.proposal_nums)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=SOURCES, default='tiny')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error('--workers不能为负数')
    prepare(args.repo.expanduser().resolve(), args.data_root.expanduser().resolve(),
            args.out.expanduser().resolve(), args.model, args.workers)


if __name__ == '__main__':
    main()
