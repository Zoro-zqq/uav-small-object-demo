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

审计同时执行文件结构检查和完整像素解码，以发现 `verify()` 可能漏检的截断图片。发现坏图会打印文件路径，并记录在 JSON 的 `corrupt_image_count` / `corrupt_image_examples`（最多 10 个示例）中；抽样拼图会在相应位置标记 `IMAGE DECODE FAILED`，不会因此中断绘图。报告保存后，审计仍以非零退出码提示数据问题，不代表数据通过验收。请从可靠原始副本重新传输并核对文件哈希，或重新下载、解压后恢复损坏文件；不要通过容忍截断图片来掩盖问题。脚本不会自动删除或修复原始图片。

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

## 新增：尺寸分组评估与 YOLO11 对比

阶段总结及待验收项目见 `BASELINE_REPORT.md`。这里的指标是自定义 `demo_original_area_ap101_v1`，不是官方 COCO 或 VisDrone 成绩。无需额外安装 pycocotools，使用 NumPy 计算。

真实框来自转换后的 YOLO 标签，按图片原始宽高还原像素坐标；面积区间为 small `[0,1024)`、medium `[1024,9216)`、large `[9216,+∞)`，不会随推理 imgsz 改变。转换时的裁边及文本小数精度会影响极靠近边界的目标，数据版本与转换结果必须固定。不要与 `analyze_small_objects.py` 的原始框统计混作同一套数量。

协议按类别和置信度排序做一对一匹配，优先匹配组内真实框；匹配组外真实框的预测忽略，未匹配且自身面积不在组内的预测也忽略，不会简单删除中大目标标签。AP 使用 IoU 0.50:0.05:0.95 和 101 个召回采样点，指标对有真实框的类别宏平均；无真实框的分组输出 null，不伪造为 0。

- `AP50_95`、`AP50`：分组检测质量，采集低置信度预测（默认 conf=0.001）。
- `AR50_95`：相同 IoU 范围、检测数上限内的最大召回率均值。
- `recall50_at_conf`：IoU=0.50、score>=0.25 的类别宏平均召回率，不等于整体 Recall 或 AR。
- NMS IoU=0.7，每图所有类别合计 max_det=300，分组推理 rect=False、无测试增强；密集场景的 300 框上限可能限制召回，改变它应建立另一套协议并重评所有模型。
- 原始 ignore 区域未进入评估，可能影响误检计数；与 Ultralytics 的积分/预处理协议有差异，分组的 all 值不要求等于原有整体 mAP，只能同协议横向比较。

在服务器项目目录运行（Linux Bash）：

```bash
python evaluate.py --weights outputs/checkpoints/visdrone_yolov8n_15ep/weights/best.pt --size-groups --imgsz 960 --batch 4 --output-dir outputs/metrics/yolov8n_15ep_size_v1
python evaluate.py --weights outputs/checkpoints/visdrone_yolov8n_30ep/weights/best.pt --size-groups --imgsz 960 --batch 4 --output-dir outputs/metrics/yolov8n_30ep_size_v1
```

每次先进行原有整体验证，再额外推理完整验证集并做 CPU 匹配，因此会比原评估慢。不要把 conf=0.25 的展示预测直接当 AP 输入。脚本不会启动训练或改变数据。输出包括：

```text
best_val_summary.json     原有整体指标及显式评估参数
best_size_summary.json    分组/分类别指标、协议、版本、数据与权重哈希
best_size_metrics.csv     四个面积组的摘要，可导入表格软件
best_size_inputs.json     实际原图坐标 GT 与预测，可复算（含数据标签，不公开上传）
```

分组输入存档可在无 GPU 环境用 `evaluate_size_groups()` 重新计算；不要编辑后与原始报告混用。数据指纹覆盖本次验证图片和 YOLO 标签的名称及内容；不证明训练集、划分间无泄漏，也不证明历史训练数据没变。重复命令会覆盖同目录的 JSON/CSV，保留历史实验时请使用新输出目录。

确认两次 YOLOv8 的分组协议、软件版本和数据指纹一致并总结失败案例后，再训练 YOLO11n，不同时改其他设置：

```bash
mkdir -p outputs/logs
set -o pipefail
python -u train.py --config configs/train_yolo11.yaml 2>&1 | tee "outputs/logs/yolo11n_30ep_$(date +%Y%m%d_%H%M%S).log"
python evaluate.py --weights outputs/checkpoints/visdrone_yolo11n_30ep/weights/best.pt --size-groups --imgsz 960 --batch 4 --output-dir outputs/metrics/yolo11n_30ep_size_v1
```

若训练输出目录带数字后缀，请使用实际权重路径。保留现有 CUDA/PyTorch，不为换模型盲目升级环境；若必须换软件版本，则在新环境重评全部待比较权重。两种 nano 模型并非参数量/耗时相等，汇报实际训练成本。注意当前 `configs/train.yaml` 可能已被手动改动，以历史 `args.yaml` 为准，不覆盖现有设置。

## 输出目录说明

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
