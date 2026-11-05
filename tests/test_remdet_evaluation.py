"""覆盖只转换验证集、评估配置隔离与结果单位，均不运行GPU。"""
import json
from pathlib import Path
import sys
from unittest.mock import patch

from PIL import Image
import pytest

from uav_detection.datasets.converters import visdrone_to_coco as converter
from uav_detection.experiments.remdet_results import METRICS, summarize


def test_val_only_conversion_does_not_require_training_data(tmp_path):
    source = tmp_path / 'raw/VisDrone2019-DET-val'
    (source / 'images').mkdir(parents=True)
    (source / 'annotations').mkdir()
    Image.new('RGB', (40, 40)).save(source / 'images/a.jpg')
    (source / 'annotations/a.txt').write_text('1,2,3,4,1,4,0,0\n')
    output = tmp_path / 'converted'
    # Windows无软链接权限时用同名目录替代，仅测试split选择和JSON，真实Linux软链接仍需服务器验证。
    with patch.object(Path, 'symlink_to', lambda path, *args, **kwargs: path.mkdir()), \
            patch.object(sys, 'argv', [
                'convert-coco', '--root', str(tmp_path / 'raw'), '--out', str(output),
                '--split', 'val', '--ignore-policy', 'drop']):
        converter.main()
    report = json.loads((output / 'conversion_report.json').read_text(encoding='utf-8'))
    assert set(report['splits']) == {'val'}
    assert not (output / 'annotations/VisDrone2019-DET_train_coco.json').exists()
    assert report['splits']['val']['counts']['kept_boxes'] == 1


def test_result_summary_converts_units_and_rejects_ambiguous_runs(tmp_path):
    run = tmp_path / '20261105_100000'
    run.mkdir()
    metrics = {key: value for key, value in zip(METRICS.values(), [.218, .371, .219, .127, .330, .445])}
    (run / '20261105_100000.json').write_text(json.dumps(metrics))
    values = summarize(tmp_path)
    assert values['AP'] == .218
    summary = json.loads((tmp_path / 'metrics_summary.json').read_text(encoding='utf-8'))
    assert summary['metrics_percent']['AP'] == 21.8
    assert '21.8' in (tmp_path / 'metrics.csv').read_text(encoding='utf-8-sig')
    with pytest.raises(ValueError, match='已存在'):
        summarize(tmp_path)
    other = tmp_path / '20261105_110000'
    other.mkdir()
    (other / '20261105_110000.json').write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match='找到2份'):
        summarize(tmp_path)


def test_eval_config_uses_matching_source_and_no_training_paths(tmp_path):
    mmengine = pytest.importorskip('mmengine.config')
    from uav_detection.experiments.remdet_eval import prepare

    repo = tmp_path / 'RemDet'
    sources = repo / 'config_remdet/remdet'
    sources.mkdir(parents=True)
    # 独立fixture同时检查继承、路径展开和来源NMS/proposal配置的保留。
    (sources / 'remdet_tiny-300e_coco.py').write_text('''
model = dict(backbone=dict(type='RemNet', widen_factor=.25, deepen_factor=.33),
             neck=dict(type='RemDetPAFPN'),
             bbox_head=dict(head_module=dict(num_classes=10)),
             test_cfg=dict(score_thr=.001, nms=dict(iou_threshold=.7)))
test_pipeline = [dict(type='LetterResize', scale=(640,640))]
test_dataloader = dict(batch_size=16, num_workers=8,
                      dataset=dict(data_root='/wrong', ann_file='wrong',
                                   data_prefix=dict(img='wrong'), indices=8))
test_evaluator = dict(type='mmdet.CocoMetric', ann_file='/wrong', proposal_nums=(100,1,10))
train_dataloader = dict(dataset=dict(data_root='/wrong'))
''', encoding='utf-8')
    data = tmp_path / 'data'
    (data / 'annotations').mkdir(parents=True)
    (data / 'images/val').mkdir(parents=True)
    (data / 'images/val/a.jpg').touch()
    (data / 'conversion_report.json').write_text(json.dumps({
        'protocol': converter.PROTOCOL, 'splits': {'val': {}}}))
    (data / 'annotations/VisDrone2019-DET_val_coco.json').write_text(json.dumps({
        'categories': [{'id': i, 'name': name} for i, name in enumerate(converter.CLASSES, 1)],
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
    }))
    target = prepare(repo, data, tmp_path / 'out', 'tiny', 0)
    cfg = mmengine.Config.fromfile(str(target))
    assert cfg.train_dataloader is cfg.train_cfg is cfg.optim_wrapper is None
    assert cfg.val_dataloader is cfg.val_cfg is cfg.val_evaluator is None
    assert cfg.test_dataloader.dataset.data_root == str(data) + '/'
    assert cfg.test_dataloader.dataset.data_prefix.img == 'images/val'
    assert 'indices' not in cfg.test_dataloader.dataset
    assert cfg.test_dataloader.persistent_workers is False
    assert cfg.test_evaluator.proposal_nums == (100, 1, 10)
    assert cfg.model.test_cfg.nms.iou_threshold == .7
    assert cfg.test_evaluator.ann_file == str(data / 'annotations/VisDrone2019-DET_val_coco.json')
