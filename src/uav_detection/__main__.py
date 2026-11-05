"""统一命令入口；各子命令仅加载自己需要的依赖。"""
from __future__ import annotations

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "audit": ("uav_detection.datasets.audit", "审计VisDrone原始标注和图片"),
    "convert": ("uav_detection.datasets.converters.visdrone_to_yolo", "VisDrone转YOLO标签"),
    "convert-coco": ("uav_detection.datasets.converters.visdrone_to_coco", "VisDrone转COCO JSON"),
    "statistics": ("uav_detection.datasets.statistics", "VisDrone/COCO数据分布统计"),
    "train": ("uav_detection.cli.train", "训练Ultralytics基线模型"),
    "evaluate": ("uav_detection.cli.evaluate", "验证模型与原图尺寸分组评估"),
    "predict": ("uav_detection.cli.predict", "模型推理与结果可视化"),
    "analyze-sizes": ("uav_detection.analysis.object_sizes", "原始目标尺寸分析"),
    "analyze-failures": ("uav_detection.analysis.failures", "复用已有预测分析失败案例"),
    "prepare-remdet": ("uav_detection.experiments.remdet", "生成外部RemDet仓库的单卡短跑配置"),
    "prepare-remdet-eval": ("uav_detection.experiments.remdet_eval", "生成官方RemDet权重的验证集评估配置"),
    "summarize-remdet": ("uav_detection.experiments.remdet_results", "整理RemDet测试指标为CSV与JSON"),
}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="uav-detect", description="无人机小目标检测：数据、实验与分析",
        epilog="子命令参数：uav-detect <command> --help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS,
                        help="\n".join(f"{key}: {value[1]}" for key, value in COMMANDS.items()))
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return
    # 其余参数交由原有argparse入口解释，避免提前吞掉子命令的--help。
    command = parser.parse_args(arguments[:1]).command
    module = import_module(COMMANDS[command][0])
    previous = sys.argv
    try:
        sys.argv = [f"uav-detect {command}", *arguments[1:]]
        module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
