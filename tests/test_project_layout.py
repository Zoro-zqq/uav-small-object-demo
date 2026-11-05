"""验证移动后的路径解析和统一入口，不运行真实模型。"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import yaml

from uav_detection.__main__ import main
from uav_detection.cli import train
from uav_detection.common.paths import project_root
from uav_detection.datasets.converters.visdrone_to_coco import convert_split, parse_row
from PIL import Image
import pytest


def test_unified_dispatch_restores_arguments():
    previous = sys.argv
    with patch('uav_detection.datasets.statistics.main') as command:
        main(['statistics', '--format', 'coco', '--output-dir', 'example'])
        command.assert_called_once_with()
    assert sys.argv is previous


def test_training_from_another_directory_keeps_workspace_paths(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    dataset = workspace / 'dataset.yaml'
    dataset.write_text('val: images\n', encoding='utf-8')
    cfg = workspace / 'experiment.yaml'
    cfg.write_text(yaml.safe_dump({
        'dataset_yaml': 'dataset.yaml', 'model': 'yolov8n.pt',
        'project': 'outputs/checkpoints', 'epochs': 20,
    }), encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    original = Path.cwd()
    try:
        os.chdir(elsewhere)
        with patch.dict(os.environ, {'UAV_PROJECT_ROOT': str(workspace)}), \
             patch.object(sys, 'argv', ['train', '--config', 'experiment.yaml', '--smoke']), \
             patch('uav_detection.cli.train.create_model') as create:
            train.main()
            kwargs = create.return_value.train.call_args.kwargs
            assert kwargs['data'] == str(dataset)
            assert kwargs['project'] == str(workspace / 'outputs/checkpoints')
            assert kwargs['epochs'] == 1 and kwargs['workers'] == 0
    finally:
        os.chdir(original)


def test_all_subcommands_help_work_outside_repository(tmp_path):
    from uav_detection.__main__ import COMMANDS
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root() / 'src')
    for command in COMMANDS:
        result = subprocess.run(
            [sys.executable, '-m', 'uav_detection', command, '--help'],
            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20,
        )
        assert result.returncode == 0, (command, result.stderr)
        assert 'usage:' in result.stdout


def test_coco_conversion_preserves_protocol_and_rejects_bad_input(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'annotations').mkdir()
    Image.new('RGB', (20, 20)).save(tmp_path / 'images/sample.jpg')
    (tmp_path / 'annotations/sample.txt').write_text(
        '1,2,4,5,1,4,0,0\n'
        '0,0,5,5,0,0,0,0\n'
        '0,0,5,5,1,11,0,0\n'
        '1,2,4,5,0,3,0,0\n'
        '-1,0,4,4,1,10,1,2\n', encoding='utf-8')
    coco, stats, excluded = convert_split(tmp_path, 'train')
    assert [ann['category_id'] for ann in coco['annotations']] == [4, 10]
    assert coco['annotations'][-1]['bbox'] == [-1, 0, 4, 4]
    assert stats['counts']['kept_boxes'] == 2 and len(excluded) == 3
    assert coco['info']['conversion_protocol'] == 'valid10_drop_ignored_v1'
    with pytest.raises(ValueError):
        parse_row('0,0,1,1,1,12,0,0', 'fixture')
    with pytest.raises(ValueError):
        parse_row('0,0,1,1', 'fixture')
