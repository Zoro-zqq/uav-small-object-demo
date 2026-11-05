"""从仓库内配置生成路径已展开的单卡短跑配置，不修改官方配置。"""
import argparse
import json
from pathlib import Path


def prepare(repo, data_root, output, batch_size, workers, lr):
    from mmengine.config import Config

    report_path = data_root / 'conversion_report.json'
    if not report_path.is_file():
        raise ValueError(f'缺少转换报告：{report_path}；先运行配套转换脚本')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('protocol') != 'valid10_drop_ignored_v1':
        raise ValueError('未识别的数据转换协议')
    if output.exists():
        raise ValueError(f'拒绝覆盖输出目录：{output}')
    sources = {
        'remdet_tiny': repo / 'config_remdet/remdet/remdet_s-300e_visdrone.py',
        'yolov8_n': repo / 'config_remdet/yolov8/yolov8_s-300e_visdrone.py',
    }
    configs = {}
    for name, source in sources.items():
        cfg = Config.fromfile(str(source))
        cfg.merge_from_dict({
            'model.backbone.deepen_factor': 0.33,
            'model.backbone.widen_factor': 0.25,
            'model.neck.deepen_factor': 0.33,
            'model.neck.widen_factor': 0.25,
            'model.bbox_head.head_module.widen_factor': 0.25,
        })
        cfg.deepen_factor = 0.33
        cfg.widen_factor = 0.25
        cfg.data_root = str(data_root) + '/'
        cfg.train_batch_size_per_gpu = batch_size
        cfg.train_num_workers = workers
        cfg.val_batch_size_per_gpu = 1
        cfg.val_num_workers = workers
        cfg.base_lr = lr
        cfg.optim_wrapper.optimizer.lr = lr
        cfg.optim_wrapper.optimizer.batch_size_per_gpu = batch_size
        cfg.optim_wrapper.accumulative_counts = 1
        for key, split, current_batch in (
            ('train_dataloader', 'train', batch_size),
            ('val_dataloader', 'val', 1),
            ('test_dataloader', 'val', 1),
        ):
            loader = cfg[key]
            loader.batch_size = current_batch
            loader.num_workers = workers
            loader.persistent_workers = workers > 0
            loader.dataset.data_root = str(data_root) + '/'
            loader.dataset.ann_file = f'annotations/VisDrone2019-DET_{split}_coco.json'
            loader.dataset.data_prefix.img = f'images/{split}'
        val_ann = data_root / 'annotations/VisDrone2019-DET_val_coco.json'
        for key in ('val_evaluator', 'test_evaluator'):
            cfg[key].ann_file = str(val_ann)
        # 保留300轮正式日程，仅让Runner运行1轮；避免把短跑伪装成缩短后的正式方案。
        cfg.train_cfg.max_epochs = 1
        cfg.train_cfg.val_interval = 1
        cfg.train_cfg.dynamic_intervals = []
        cfg.default_hooks.param_scheduler.warmup_epochs = 10
        cfg.default_hooks.checkpoint.interval = 1
        cfg.default_hooks.logger.interval = 20
        cfg.randomness = dict(seed=42, deterministic=False)
        cfg.load_from = None
        cfg.resume = False
        cfg.work_dir = str(repo / 'work_dirs' / (name + '_smoke_v100'))
        configs[name] = cfg
    # 两个配置均读取成功后再写出，来源配置始终不变。
    output.mkdir(parents=True)
    for name, cfg in configs.items():
        target = output / (name + '_smoke.py')
        cfg.dump(str(target))
        reread = Config.fromfile(str(target))
        assert reread.model.backbone.widen_factor == 0.25
        assert reread.train_dataloader.dataset.data_root == str(data_root) + '/'
        assert reread.val_evaluator.ann_file == str(val_ann)
        print('配置：', target)
    manifest = {
        'purpose': '1-epoch smoke only; not a final paper reproduction recipe',
        'conversion_protocol': report['protocol'],
        'batch_size': batch_size, 'accumulative_counts': 1,
        'smoke_lr': lr, 'runner_epochs': 1,
        'scheduler_epochs': 300, 'seed': 42,
        'note': '保留来源配置的评价设置；正式lr/batch/epochs需根据日志及测速确定。',
        'sources': {name: str(path) for name, path in sources.items()},
    }
    (output / 'smoke_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke-lr', type=float, default=0.001)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.out.expanduser().resolve() if args.out else repo / 'configs_local_v100'
    if args.batch_size < 1 or args.workers < 0 or args.smoke_lr <= 0:
        raise ValueError('batch_size、workers或lr不合法')
    prepare(repo, args.data_root.expanduser().resolve(), output,
            args.batch_size, args.workers, args.smoke_lr)


if __name__ == '__main__':
    main()
