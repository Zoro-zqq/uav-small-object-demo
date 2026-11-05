# 单V100：原生VisDrone转COCO与RemDet起步

> 2026-11-05目标更新：用户当前只需要官方权重评估，使用[从安装到结果导出的评估指南](remdet-official-eval.md)。本页保留为后续训练参考；用户最新原生数据路径为`/home/lx/data/zqq/VisDroneDataset`，本项目服务器路径已确认为`/home/lx/data/zqq/code/uav-small-object-demo`。

2026-11-05。服务器RemDet仓库：`/home/lx/data/zqq/code/RemDet`；原生数据：`/home/lx/data/zqq/VisDrone2019-DET`。本项目不含模型权重或数据，不修改官方代码与原始数据。

## 1. 同步项目和环境

把本项目代码同步到服务器，与RemDet仓库并列放置即可。下面推荐本项目放在`/home/lx/data/zqq/code/uav-small-object-demo`；这不是已经核实的服务器路径，实际不同就修改cd命令。转换器现位于`src/uav_detection/datasets/converters/visdrone_to_coco.py`，配置工具位于`src/uav_detection/experiments/remdet.py`，无需向RemDet根目录复制脚本。

激活已经安装RemDet的独立环境：

```bash
cd /home/lx/data/zqq/code/uav-small-object-demo
conda activate remdet-repro
python -m pip install -e . --no-deps
uav-detect convert-coco --help
```

前提是RemDet独立环境已安装Pillow、PyYAML、NumPy及MMEngine等依赖；`--no-deps`避免为安装本工具意外更新该框架环境。若尚未准备环境，先按Obsidian笔记中的《服务器复现起步》完成安装和CUDA算子检查；环境名称若不同，使用实际名称。数据工具和RemDet训练通过绝对路径对接。

## 2. 转换数据

```bash
uav-detect convert-coco \
  --root /home/lx/data/zqq/VisDrone2019-DET \
  --out /home/lx/data/zqq/VisDrone2019-DET-COCO-remdet-v1 \
  --ignore-policy drop
```

支持常见的`VisDrone2019-DET-train/images`、`VisDrone2019-DET-val/images`，或`train/images`、`val/images`布局，每个split还需同级`annotations/*.txt`。自动发现不明确时脚本会报错，此时显式追加`--train-dir 实际训练目录 --val-dir 实际验证目录`，不猜测。拒绝覆盖已有输出目录。

输出包括：

```text
VisDrone2019-DET-COCO-remdet-v1/
├── images/train → 原训练图片目录（软链接）
├── images/val   → 原验证图片目录（软链接）
├── annotations/VisDrone2019-DET_train_coco.json
├── annotations/VisDrone2019-DET_val_coco.json
├── annotations/train_excluded_rows.json
├── annotations/val_excluded_rows.json
└── conversion_report.json
```

数据处理协议为`valid10_drop_ignored_v1`：保留类别1–10且score=1的有效框，类别名与仓库一致；排除类别0、11、score=0、宽高小于1及完全不与图片相交的框，并逐条记录；不因遮挡/截断等级额外过滤，不裁图、不调整原坐标、不主动裁剪跨边界框。没有有效框的图片仍保留。脚本检查8列原生真值格式、图片与TXT一一对应，遇到未知格式明确报错。

**协议边界**：排除忽略标注不等于在训练或评价中自动屏蔽这些图像区域；相应预测仍可能被COCO评价记为误报。COCO的iscrowd规则也不能直接视作VisDrone原生忽略规则。本仓库尚未找到已核实的作者转换脚本或转换后JSON，因此本转换用于明确协议下的起步与同协议对照，绝对AP不保证可直接对齐论文。原始标注完整保留，后续取得作者协议时可重新生成新版本，不覆盖本版本。

常见完整划分为训练6471、验证548张；以实际统计核对，数量异常需查原因，而非强行补齐。转换完成后请先把终端统计输出贴回来。

## 3. 生成单卡短跑配置

环境安装和数据转换成功后：

```bash
uav-detect prepare-remdet \
  --repo /home/lx/data/zqq/code/RemDet \
  --data-root /home/lx/data/zqq/VisDrone2019-DET-COCO-remdet-v1 \
  --batch-size 4 --workers 2
```

输出到仓库`configs_local_v100/`：

- `remdet_tiny_smoke.py`：完整RemDet，宽度0.25、深度0.33。
- `yolov8_n_smoke.py`：同仓库YOLOv8基线，宽度0.25、深度0.33。
- `smoke_manifest.json`：这次短跑的设置记录。

生成器读取并展开官方S配置，更新模型嵌套宽深度、所有数据集路径、评价JSON路径、真实batch及优化器batch字段；不修改来源配置。默认短跑学习率0.001、训练1轮、无梯度累积、seed42，验证batch1。300轮的学习率日程保留，目的为连通性与测速，**不是1轮正式训练配方，也不是最终复现超参数**。已消除两来源配置不同初始lr/预热设置对这次短跑的直接差异；不声称小batch能严格等价原文多卡设置。继承的CocoMetric proposal_nums等评价参数原样保留，正式评价协议仍需核查。

## 4. 先运行RemDet短跑

配置生成完成后，进入外部RemDet仓库运行它自己的训练入口：

```bash
cd /home/lx/data/zqq/code/RemDet
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs_local_v100/remdet_tiny_smoke.py \
  --amp --work-dir work_dirs/remdet_tiny_smoke_v100
```

跑完或报错后，把开始时环境/配置摘要、训练的`time/data_time/memory/loss`和最后验证指标贴回来。中途出现显存不足就停止本次进程，记录报错；可以用`--batch-size 2 --out configs_local_v100_bs2`重新生成一套配置再试，不覆盖已有日志。AMP使用FP16，V100不采用BF16。

只有RemDet短跑成功后，再顺序运行基线短跑：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs_local_v100/yolov8_n_smoke.py \
  --amp --work-dir work_dirs/yolov8_n_smoke_v100
```

一轮AP只说明管线能运行，不能用于判定谁更好。根据两者实际耗时、显存与截止日期，再确定正式epoch、学习率日程、Mosaic关闭时点和是否采用梯度累积；正式训练使用新work_dir且不自动继承短跑权重。

## 本地验证范围

已做语法、合成数据和原有回归检查；这些检查不等于服务器环境、真实数据转换或GPU训练已成功。初始辅助脚本验证见[验证记录](../reports/remdet-tools-validation.md)，目录重构后的检查见[迁移说明](../architecture/migration-2026-11-05.md)。
