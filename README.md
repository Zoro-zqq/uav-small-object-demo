# 无人机小目标检测实验工作区

用于复用数据审计、标签转换、基线训练、评价和失败分析。当前支持Ultralytics YOLOv8/YOLO11基线、VisDrone→YOLO/COCO转换、VisDrone/COCO统计，以及外部RemDet仓库的短跑配置、官方权重评估配置与指标汇总。

## 目录职责

```text
src/uav_detection/       可复用代码与统一命令入口
├── common/             YAML与工作区路径
├── datasets/           数据集解析、审计、统计
│   └── converters/     按输入/输出格式命名的标签转换器
├── backends/           模型框架适配，目前为Ultralytics
├── cli/                训练、评价、预测
├── evaluation/         指标协议与原图尺寸分组评价
├── analysis/           尺寸、失败案例及图表分析
└── experiments/        外部模型仓库配置适配，目前为RemDet
configs/
├── datasets/           可提交的示例与本机*.local.yaml
└── experiments/        按数据集/模型组织的实验配方
docs/
├── guides/             操作说明与数据准备
├── reports/            人工撰写的实验报告
├── learning/           已有学习记录与概念笔记
├── plans/              研究计划与验收标准
└── architecture/       扩展约定与迁移说明
outputs/                已有及新生成的训练/评价产物
tests/                  数据、指标与入口回归检查
```

原始图片、标注和第三方模型仓库放在项目外。`outputs/`的历史文件保持原位；未来新产物默认不提交，精选报告放`docs/reports/`。论文阅读笔记仍使用已有Obsidian文库，本项目不再建立第二套论文库。

## 安装与入口

在项目根目录选择所需依赖安装，不为统计工具安装整套GPU框架：

```bash
python -m pip install -e .
python -m pip install -e ".[ultralytics,dev]"
uav-detect --help
```

第二条适用于需要YOLO基线的独立环境；已有CUDA/PyTorch按原环境保留。RemDet自身仍使用其独立环境和官方安装方式，`prepare-remdet`只生成配置，需要该环境可导入MMEngine。

也可使用`python -m uav_detection <子命令>`。包需先安装；源码测试可直接`python -m pytest -q`。

## 常用流程

先将`configs/datasets/visdrone.example.yaml`复制为`visdrone.local.yaml`并填写实际路径。本次目录整理已保留现有本机配置内容，没有改写其数据根目录。

```bash
uav-detect audit --split all --visualize 20
uav-detect convert --split all
uav-detect train --config configs/experiments/visdrone/yolov8n.yaml --smoke
uav-detect train --config configs/experiments/visdrone/yolov8n.yaml
uav-detect evaluate --weights outputs/checkpoints/实际实验/weights/best.pt --size-groups
```

现有YOLOv8配方保留20轮、YOLO11配方保留30轮，这是原有设置，**不是同预算对照**；要做比较需显式对齐轮数及其他训练条件，历史训练以`outputs/checkpoints/*/args.yaml`为准。命令通过`--model`或实验YAML选择模型，勿仅改文件名便声称运行了另一模型。

只使用RemDet官方权重时，按[官方权重评估指南](docs/guides/remdet-official-eval.md)安装环境、只转换val、生成配置并运行官方`tools/test.py`；`uav-detect summarize-remdet`整理输出为CSV。需要训练时再看[服务器转换与短跑指南](docs/guides/remdet-server.md)。RemDet不是Ultralytics后端，不能直接用`uav-detect train --model remdet`运行。

## 文档导航

- [完整数据审计、YOLO转换、训练及评价流程](docs/guides/workflow.md)
- [AI-TOD准备与描述性统计](docs/guides/ai-tod.md)
- [RemDet服务器转换与短跑](docs/guides/remdet-server.md)
- [RemDet官方权重评估：从安装到CSV](docs/guides/remdet-official-eval.md)
- [已有基线报告](docs/reports/baseline.md)与[失败分析](docs/reports/failure-analysis.md)
- [学习验收](docs/learning/acceptance.md)、[概念速查](docs/learning/knowledge-notes.md)、[研究计划](docs/plans/a1-a3-acceptance.md)
- [如何扩展数据集、转换器和模型](docs/architecture/layout.md)
- [旧路径与命令迁移表](docs/architecture/migration-2026-11-05.md)

`demo_original_area_ap101_v1`等已有自定义指标保持原定义。常规YOLO/COCO转换不自动实现VisDrone原生忽略区域协议；不得将历史demo结果或当前转换后的AP直接称为官方成绩。
