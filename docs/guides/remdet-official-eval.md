# 单V100评估RemDet官方权重：安装到结果导出

更新：2026-11-05。当前目标是**官方权重评估**，不训练、不跑YOLOv8基线、不做训练消融。以下命令在服务器的Bash终端逐段执行，一段成功后再执行下一段。

服务器路径（用户已确认）：

- 本项目：`/home/lx/data/zqq/code/uav-small-object-demo`
- 官方仓库：`/home/lx/data/zqq/code/RemDet`
- 原生数据：`/home/lx/data/zqq/VisDroneDataset`
- GPU：Tesla V100 16GB；先使用RemDet-Tiny、单卡、batch=1、FP32。

本地核查依据为官方仓库commit `8cf2667e9ff80122e89436a0ebc6558ff1cfdb93`。本指南和工具在本机验证了配置/数据/汇总逻辑，尚未在用户服务器上执行CUDA评估。

## 0. 再同步本次新增文件

此前目录重构的同步不包含本次新增的评估工具。请再次同步本项目的`src/`、`configs/environments/`及`docs/`；保留服务器已有数据、权重和outputs。不用重新下载RemDet。

```bash
export UAV_PROJECT=/home/lx/data/zqq/code/uav-small-object-demo
export REMDET=/home/lx/data/zqq/code/RemDet
export VISDRONE_RAW=/home/lx/data/zqq/VisDroneDataset
export VISDRONE_COCO=/home/lx/data/zqq/VisDroneDataset-COCO-eval-v1
export CONSTRAINTS="$UAV_PROJECT/configs/environments/remdet-v100-eval-constraints.txt"
cd "$REMDET"
nvidia-smi
git rev-parse HEAD
find "$VISDRONE_RAW" -maxdepth 3 -type d -name images
```

仓库来自ZIP时没有git信息，可记录来源与下载日期；不要为对齐版本而直接重置现有仓库。后续命令使用这些变量；重开终端后要重新设置。

## 1. 创建独立环境

不要在原Python3.13环境继续堆依赖。若同名环境已经存在，先激活检查，不覆盖。

```bash
conda create -n remdet-eval python=3.11 pip -y
conda activate remdet-eval
python -m pip install -c "$CONSTRAINTS" setuptools wheel
python -m pip install -c "$CONSTRAINTS" torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c "$CONSTRAINTS" -r "$REMDET/requirements/runtime.txt" mmengine timm albumentations Pillow PyYAML gdown
python -m pip install -c "$CONSTRAINTS" mmcv==2.2.0 --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
cd "$REMDET"
python -m pip install -v -e . --no-build-isolation --no-deps
cd "$UAV_PROJECT"
python -m pip install -e . --no-build-isolation --no-deps
python -m pip check
uav-detect prepare-remdet-eval --help
```

这里修正了作者README中`pip install -y pytorch`的错误写法。安装的是本地RemDet仓库提供的`mmdet`实现，不要再安装上游`mmdet`覆盖它，也无需Ultralytics。固定NumPy1.26.4及OpenCV版本避免旧框架被依赖解析升级到NumPy2。MMCV使用带CUDA算子的预编译包，不能换成`mmcv-lite`。

该组合采用作者README指定的Torch2.2.0；作者公开Tiny训练日志为Torch2.3.1，因此不是作者原环境的逐包复制。已核查上述MMCV索引提供Python3.11、Linux x86_64的2.2.0 wheel。`nvidia-smi`上的CUDA版本不要求等于cu121；此处使用wheel自带运行库，无需先改系统CUDA或安装nvcc。

## 2. 检查CUDA与本地模型注册

```bash
cd "$REMDET"
python - <<'PY'
import os
import sys
from pathlib import Path
import torch, torchvision, mmcv, mmengine, mmdet
from mmcv.ops import nms
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
register_all_modules(init_default_scope=True)
print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('MMCV:', mmcv.__version__, 'MMEngine:', mmengine.__version__)
print('mmdet:', mmdet.__file__)
assert Path(mmdet.__file__).resolve().is_relative_to(Path(os.environ['REMDET']).resolve())
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
assert nms(boxes, scores, 0.5)[1].tolist() == [0]
assert MODELS.get('RemNet') is not None
assert MODELS.get('RemDetPAFPN') is not None
print('CUDA NMS和RemDet模块注册通过')
PY
```

若出现`mmcv._ext`、undefined symbol或CUDA算子错误，在这里停止并保留完整错误；不要继续转换或跑全量评估。

## 3. 下载匹配的VisDrone官方权重

权重入口来自[官方README的VisDrone结果表](https://github.com/HZAI-ZJNU/RemDet#main-results)。默认先用[RemDet-Tiny官方权重](https://drive.google.com/file/d/1oN4FrKJbOh62KhrPzyIoZwWoBlRVhvrr/view)。

```bash
cd "$REMDET"
mkdir -p checkpoints
export CKPT="$REMDET/checkpoints/remdet_tiny_visdrone.pth"
if [ ! -e "$CKPT" ]; then
  gdown 'https://drive.google.com/uc?id=1oN4FrKJbOh62KhrPzyIoZwWoBlRVhvrr' -O "$CKPT"
fi
python - <<'PY'
import os
from pathlib import Path
import torch
p = Path(os.environ['CKPT'])
c = torch.load(p, map_location='cpu')
assert isinstance(c, dict) and 'state_dict' in c, '下载文件不是预期MMEngine权重'
assert c['state_dict'], 'state_dict为空'
print('权重大小(MiB):', round(p.stat().st_size / 1024**2, 2))
print('meta keys:', list(c.get('meta', {})))
print('state_dict tensors:', len(c['state_dict']))
PY
sha256sum "$CKPT"
```

已有匹配权重时直接将`CKPT`设为真实绝对路径，不重复下载。若Google Drive不可达或限流，可在能下载的电脑上手动下载后上传到此路径，不能把网页HTML改名成`.pth`。这里未代你下载或验证该权重文件。

**型号与数据集必须匹配**：Tiny权重使用`--model tiny`；S/M/L/X分别使用对应型号及VisDrone权重，不能加载COCO80类或UAVDT权重。Tiny配置虽然叫`remdet_tiny-300e_coco.py`，当前仓库实际继承VisDrone配置，生成器已按内容检查十类。

## 4. 只转换验证集

本次不需要训练集。标准输入应类似`VisDroneDataset/VisDrone2019-DET-val/{images,annotations}`或`VisDroneDataset/val/{images,annotations}`，图片和原生八列TXT对应。

```bash
cd "$UAV_PROJECT"
uav-detect convert-coco \
  --root "$VISDRONE_RAW" \
  --out "$VISDRONE_COCO" \
  --split val --ignore-policy drop
```

自动发现失败时，根据第0步find结果追加`--val-dir /真实路径/VisDrone2019-DET-val`，该目录下必须同时有`images/`和`annotations/`。转换器拒绝覆盖已有目录；已成功生成且协议/源数据一致时复用，失败残留目录或不同协议请另选新输出路径，勿直接删除原数据。

预计VisDrone2019-DET验证集为548张，以实际输出核对；转换输出：

```text
VisDroneDataset-COCO-eval-v1/
├── images/val -> 原验证图片目录
├── annotations/VisDrone2019-DET_val_coco.json
├── annotations/val_excluded_rows.json
└── conversion_report.json
```

**评价边界**：协议`valid10_drop_ignored_v1`保留类别1—10有效框、保留原xywh，不裁边、不减1，排除类别0、11、score0及无效框。排除ignore标注不等于屏蔽ignore区域，COCO评价可能把其中预测计为误报。尚未取得作者转换脚本/JSON，因此不能保证AP与论文完全对齐；先如实报告本次转换协议。

## 5. 生成本次评估配置并严格核对权重形状

```bash
export RUN="$UAV_PROJECT/outputs/evaluation/remdet_tiny_official_$(date +%Y%m%d_%H%M%S)"
uav-detect prepare-remdet-eval \
  --repo "$REMDET" --data-root "$VISDRONE_COCO" \
  --model tiny --workers 2 --out "$RUN"
mkdir -p "$RUN/smoke" "$RUN/full" "$RUN/environment"
python -m pip freeze > "$RUN/environment/pip-freeze.txt"
nvidia-smi > "$RUN/environment/nvidia-smi.txt"
git -C "$REMDET" rev-parse HEAD > "$RUN/environment/remdet-commit.txt"
sha256sum "$CKPT" > "$RUN/environment/checkpoint-sha256.txt"
printf '%s\n' "$CKPT" > "$RUN/environment/checkpoint-path.txt"
```

配置使用batch1和官方模型/测试pipeline/阈值，关闭训练与验证循环的数据入口，避免寻找训练集。保留官方`proposal_nums=(100,1,10)`，开启逐类AP和检测JSON导出；不额外开启TTA。MMCV/pycocotools会处理该maxDets列表，勿为了追求更高分擅自改成更多框。

官方测试入口默认非严格加载，故正式运行前额外检查全部模型tensor形状与键：

```bash
cd "$REMDET"
python - <<'PY'
import os
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_state_dict
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
cfg = Config.fromfile(str(Path(os.environ['RUN']) / 'eval_config.py'))
model = MODELS.build(cfg.model)
checkpoint = torch.load(os.environ['CKPT'], map_location='cpu')
state = checkpoint['state_dict']
# 官方MMEngine可能存有DDP的module前缀，与其默认加载行为一致地去除。
if state and all(key.startswith('module.') for key in state):
    state = {key[7:]: value for key, value in state.items()}
load_state_dict(model, state, strict=True)
print('权重与完整RemDet模型严格匹配')
PY
```

检查失败先核对型号、类别和仓库版本，不以`strict=False`忽略差异后继续汇报。权重里的`state_dict`按MMEngine EMAHook保存约定已用于部署；不要未经核实自行换用`ema_state_dict`。

## 6. 先用8张图跑通

```bash
cd "$REMDET"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  "$RUN/eval_config.py" "$CKPT" \
  --work-dir "$RUN/smoke" \
  --cfg-options test_dataloader.dataset.indices=8 \
                test_evaluator.outfile_prefix="$RUN/smoke/detections" \
  2>&1 | tee "$RUN/smoke/console.log"
```

`indices=8`只限制这次试跑，不修改保存的配置。成功标准是8张图推理结束且正常输出指标，**这8张的AP不用于汇报**。没有GUI的服务器不加`--show`，本测试入口没有`--amp`选项。遇到DataLoader共享内存/worker错误可以给命令追加`test_dataloader.num_workers=0 test_dataloader.persistent_workers=False`后重试；保留第一次错误日志。

## 7. 全量评估和导出

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  "$RUN/eval_config.py" "$CKPT" \
  --work-dir "$RUN/full" \
  --out "$RUN/full/predictions.pkl" \
  2>&1 | tee "$RUN/full/console.log"
uav-detect summarize-remdet --run-dir "$RUN/full"
```

确认上一条测试命令退出成功后再运行汇总。全量命令**不包含`indices=8`**；核对测试日志中图像/迭代数，应覆盖全部548张（实际数以转换报告为准）。`--out`存的是预测pkl，不是指标；MMEngine自动在时间戳目录写指标JSON，汇总工具据此生成CSV/JSON，不重新计算AP。

产物位置：

```text
outputs/evaluation/remdet_tiny_official_时间戳/
├── eval_config.py                实际评估配置
├── eval_manifest.json            模型、预处理、评价协议及样本数
├── detections.bbox.json          全量COCO格式检测结果
├── environment/                 版本、显卡、权重路径和哈希
├── smoke/                       8张图连通性检查
└── full/
    ├── console.log              完整日志及逐类AP
    ├── predictions.pkl          MMDetection预测对象
    ├── metrics.csv              AP/AP50/AP75/AP_s/AP_m/AP_l，双单位
    ├── metrics_summary.json     汇总及原指标路径
    └── 时间戳/                  框架保存的日志和原始指标JSON
```

逐类表是AP，不是某个固定置信度下的Precision；源码返回键中的`*_precision`命名不能改变其实际计算含义。本流程不自动产生固定阈值P/R/F1或可信FPS。test日志的`time`含管线开销，不能直接倒数后与论文4090上的FPS比较。

## 8. 怎样写入汇报

官方README的Tiny参考为AP21.8、AP50 37.1、AP75 21.9、AP_s12.7、AP_m33.0、AP_l44.5，均为百分数表示。上述工具不会用参考值填充结果；所有实测数据以服务器test输出为准。原日志`0.218`应写成表格中的`21.8`，不要混用单位。

可写：“在单张Tesla V100上加载作者公开的RemDet-Tiny VisDrone权重，采用官方640单尺度测试预处理，在本地转换的VisDrone2019-DET验证集上评估；转换协议为valid10_drop_ignored_v1。报告AP/AP50/AP75/AP_s等实测指标。未开展从头训练复现。”

分数异常先核对完整548张、模型/权重型号、十类顺序、预处理与忽略区域协议，不通过修改score_thr、NMS或maxDets去追平论文。下一步向协作者提供`full/metrics.csv`、`eval_manifest.json`与测试日志末尾即可分析结果。
