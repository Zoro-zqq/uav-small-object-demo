# VisDrone 无人机小目标检测 Demo

此项目先用 VisDrone2019-DET 的 train/val 数据，完成原始标注审计、YOLO 标签转换和 YOLOv8 基线。数据集文件留在独立数据目录；项目不复制、修改或提交原始图片和标注。

## 已确认的本机数据布局

```text
D:/研究/无人机数据/
├─ VisDrone2019-DET-train/
│  ├─ images/
│  ├─ annotations/    # 原始数据，只读
│  └─ labels/         # 转换脚本新建的 YOLO 标签
└─ VisDrone2019-DET-val/
   ├─ images/
   ├─ annotations/    # 原始数据，只读
   └─ labels/         # 转换脚本新建的 YOLO 标签
```

对应的配置位于 `configs/dataset.yaml`。如果把代码移到远程 GPU，必须将 `path` 改为**服务器本机数据的绝对路径**；Windows 的 `D:/...` 路径不能直接用于 Linux 服务器。

## 原始标注和忽略区域

VisDrone 每个原始标注行包含八列：`left, top, width, height, score, category, truncation, occlusion`。类别为 `1..10`，YOLO 类别从 `0` 开始；框转换成 `class_id center_x center_y width height`，中心点与宽高按原图尺寸归一化。对于 VisDrone 的忽略区域（`score=0` 或类别 `0`），转换时不输出训练框；保留原始 `annotations` 不动。

常规 YOLO 指标不是 VisDrone 官方评测：常规标签转换会排除忽略区域，而默认验证不一定应用 VisDrone 忽略区域评测协议。报告应标明这是经转换数据上的 demo 基线；不要将其称作官方 benchmark 分数。严格官方成绩需使用兼容的官方 evaluator 和完整忽略区域处理。

## 安装

优先在与训练服务器分离的虚拟环境中安装通用工具。若远端已有 PyTorch/CUDA 环境，先保留其 torch/torchvision 组合，再安装 demo 依赖；`requirements.txt` 不指定 torch 版本。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

确认训练环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import ultralytics; print(ultralytics.__version__)"
```

## 第一阶段：原始数据审计

```powershell
python data_audit.py --split all --visualize 20
```

程序检查：图片与标注匹配、可读性、原始八字段、类别与 score 分布、越界框、类别分布、目标大小分组，并保存随机可视化图。审计 JSON 写入 `outputs/metrics/data_audit_summary.json`；标注可视化图写入 `outputs/predictions/`。对超出图像边界的框会报告供人工复核；转换时只裁剪少量越界部分，并对完全位于边界外的框报错。

查看 `outputs/metrics/data_audit_summary.json`，并人工核对抽样图里的框是否和目标大致对齐。首次学习时只跑这个阶段，观察框、`score`、类别字段和图像尺寸。

## 第二阶段：转换 YOLO 标签

```powershell
python convert_annotations.py --split all
```

新标签分别写入数据集 split 下的 `labels/`，并生成 `configs/visdrone.yaml` 供 Ultralytics 使用。默认不覆盖已有标签；若想重跑覆盖**同名生成标签**，需要明确添加 `--overwrite`。原始图片、原始标注和目录中的其他文件不会删除。转换摘要写入 `outputs/metrics/annotation_conversion_summary.json`。

转换后随机检查至少 20 张图，并确认 YOLO txt 内容符合：

```text
0 0.52 0.41 0.08 0.13
```

五列依次是类别 ID、归一化中心 x/y、归一化宽/高；空标签文件表示图片没有保留的有效目标。

## 第三阶段：先冒烟训练

```powershell
python train.py --smoke
```

冒烟模式仅抽取约 0.5% 训练图片、跑 1 epoch，以确认配置、标签、模型下载/加载和 GPU 前向反向链路；它不是模型性能实验。如果训练环境离线或无法下载官方权重，先准备获准的本地预训练权重，再通过 `--model` 指定本地路径。

通过后再运行正式训练：

```powershell
python train.py --config configs/train.yaml
```

训练默认模型 `yolov8n.pt`、图片尺寸 `960`、batch `4`、epochs `100`、GPU `0`，所有参数可以用 CLI 覆盖。显存不足时先减小 `batch`，再评估 `imgsz`。正式训练前要保存配置副本，并确认所用 Ultralytics 权重、软件及数据许可满足研究、演示与后续使用要求。

## 评估、尺寸分析和预测

```powershell
python evaluate.py --weights outputs/checkpoints/visdrone_yolov8n/weights/best.pt
python analyze_small_objects.py --split all
python predict.py --weights outputs/checkpoints/visdrone_yolov8n/weights/best.pt --source D:/研究/无人机数据/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg
```

验证默认使用 train/val 中的转换标签。`test-dev` 不参与反复调参；最终 report 需列明数据转换、忽略区域处理、软件/权重、超参、指标与局限。

## 输出目录

```text
outputs/
├─ checkpoints/    # 训练日志、配置和模型权重
├─ failures/       # 后续存放失败案例
├─ metrics/        # JSON 审计、转换、指标与尺寸统计
└─ predictions/    # 随机标注图与推理可视化
```

默认忽略数据、权重、训练结果、生成标签和本地机器路径；不要把 VisDrone 数据或学术用途受限的数据推送到公开仓库。

## 数据与软件引用

- 数据与许可要求：[VisDrone 官方仓库](https://github.com/VisDrone/VisDrone-Dataset)、[官方数据页](https://visdrone.org/visdrone/data-base)。下载、引用、展示和再分发前按当前官方条款核查。
- 检测工具：[Ultralytics](https://docs.ultralytics.com/)；使用其预训练模型、软件或服务前核实对应版本及其 AGPL/Enterprise 许可和项目用途是否相符。
- 使用 PyTorch 时按远端既有 CUDA 环境选择兼容版本；不要为此升级系统驱动。
