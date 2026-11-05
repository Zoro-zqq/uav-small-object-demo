# AI-TOD原版：下载、整理与三类数据统计

日期：2026-11-03。对应用户给出的[jwwangchn/AI-TOD](https://github.com/jwwangchn/AI-TOD)。先做描述性统计，不训练模型，不需要GPU。新增入口为[src/uav_detection/datasets/statistics.py](../../src/uav_detection/datasets/statistics.py)，使用项目已有Pillow、numpy、matplotlib与PyYAML。

最新状态：用户已把数据下载到`D:/研究/无人机数据/AI-TOD`，无需重复生成图片。本地结构为`annotations/aitod_train_v1.json`、`annotations/aitod_val_v1.json`以及`train/images`、`val/images`，不是下文教学示例的`images/train`。已完成真实train/val统计和三类训练集对照，见[结果与检查限制](../../outputs/dataset_statistics/train_comparison_v1/summary.md)。以下通用下载流程保留备查，当前直接使用已有数据。

本地labels目录抽样为`xmin ymin xmax ymax 类别名`，不是YOLO标签；此次用COCO JSON统计，没有转换或覆盖它们。train/val分别有5474/1332个越界框，统计保留原尺寸；图片存在/头部尺寸检查无失败，但未完成完整解码、全量标签对应和泄漏审计。后续训练前需明确这些处理规则。

## 1. 官方当前提供什么

本次通过GitHub API读取[官方README](https://github.com/jwwangchn/AI-TOD/blob/master/README.md)，其blob SHA为`dc46b9aa9bbf95ba74ebf54810609a16a5986550`。该版本明确写明：完整标注已提供，但完整图像需要把两部分数据结合，用作者工具生成。

| 获取项 | 官方入口 | 用途 |
|---|---|---|
| AI-TOD完整标注及不含xView的图片部分 | [README当前公开的Google Drive目录](https://drive.google.com/drive/folders/1uNY_rcOO5LrWibXRY6l2dvqSbK6xikJp?usp=sharing) | 从中获取标注JSON，选择明确的train/val文件；下载图片部分以便后续补齐 |
| xView训练数据 | [xView官网](http://xviewdataset.org/#dataset) | 官方生成流程需要训练图像TIFF及对应`xView_train.geojson`；获取条件以网站为准 |
| 生成工具 | [aitodtoolkit](https://github.com/jwwangchn/AI-TOD/tree/master/aitodtoolkit) | 用指定文件列表生成缺少的xView来源图片，组合成AI-TOD |
| 官方评测实现 | [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod) | 后续复现论文指标时再核对使用；三类统计不需要安装 |

`git clone`只下载仓库和工具，不等于完整数据集下载。当前仓库`aitodtoolkit/aitod/annotations/`里仅有标注已发布的提示文件，没有实际标注JSON；JSON需从README所指数据入口获取。README源码中被注释的历史网盘链接不作为当前推荐入口。本次未下载Google Drive/xView大文件，也未验证网盘内文件完整性或你的访问权限。

先固定为用户链接的AI-TOD原版，记录下载来源和实际标注文件名。不要将AI-TOD-v2的重新标注与原版数字混合；如果采用v2，应另建版本记录，并同步调整全部基线和评测说明。

## 2. 推荐先做：下载标注，立即完成统计

1. 打开官方README的Google Drive链接，先下载完整标注包，解压到独立数据目录。
2. 找出train和val的JSON。查看其顶层是否有`images`、`annotations`、`categories`；不要只凭压缩包名称判断完整。
3. 本轮以两数据集各自的train划分作主要分布对照，val单独统计。保留官方划分，不混合train、val、trainval、test，不把trainval再叠加train/val。
4. 直接用下节命令统计。只有标注也能计算原图目标框尺寸、每图目标数、类别长尾；图片数量来自JSON的images列表，此时不证明图片已齐全、可解码或标签已目视对齐。

以下是推荐整理结构，实际JSON名称以下载内容为准，不强制重命名：

```text
AI-TOD/
├─ annotations/
│  ├─ 实际的train标注.json
│  └─ 实际的val标注.json
└─ images/                  # 可以稍后补齐
   ├─ train/
   └─ val/
```

官方完整生成流程还涉及trainval、test目录；上图只是本轮所需范围，不表示应删掉其他划分。

## 3. 在项目根目录运行统计

先把代码同步到服务器；以下示例路径必须替换成实际路径。新脚本的CLI相对路径均相对于当前工作目录，因此请在项目根目录运行。

Linux服务器：先确认实际JSON文件名，例如`ls /data/zqq/AI-TOD/annotations/*.json`，然后运行：

```bash
uav-detect statistics \
  --format coco \
  --annotations /data/zqq/AI-TOD/annotations/实际train文件名.json \
  --name AI-TOD-original \
  --split train \
  --output-dir outputs/dataset_statistics/aitod_train_v1
```

Windows PowerShell也可以直接统计，不需要上传GPU服务器：

```powershell
uav-detect statistics --format coco --annotations "D:/研究/无人机数据/AI-TOD/annotations/实际train文件名.json" --name AI-TOD-original --split train --output-dir outputs/dataset_statistics/aitod_train_v1
```

`--split train`只记录划分名称，实际COCO数据由`--annotations`指定的JSON决定，它不会帮你把一个JSON拆成训练/验证集。要统计val，请同时换成val JSON、`--split val`和新输出目录。

VisDrone使用原始八列标注，服务器上运行：

```bash
uav-detect statistics \
  --format visdrone \
  --config configs/datasets/visdrone.local.yaml \
  --split train \
  --output-dir outputs/dataset_statistics/visdrone_train_v1
```

这里的VisDrone配置必须指向该机器实际原始数据根目录；在Windows可使用已指向本地数据的`configs/datasets/visdrone-yolo.local.yaml`。本轮在本地用此配置生成VisDrone统计，只读取图片头部和原标注，不覆盖训练标签或既有评估。

新脚本拒绝覆盖已存在的输出目录；重新统计可使用`*_v2`。输出包括：

| 文件 | 看什么 |
|---|---|
| `summary.json` | 图片/有效实例数、尺寸均值中位数、每图实例统计、类别数量、异常计数及检查范围 |
| `object_sizes.png` | 目标框等效边长sqrt(w×h)的直方分箱；两数据集使用相同箱边界和百分比纵轴 |
| `instances_per_image.png` | 每图有效实例数量分布，包含零目标图片；比较密度 |
| `class_long_tail.png` | 各自按频数排序的类别分布，纵轴为实例数对数刻度；比较类别不平衡 |
| `per_image_counts.csv`、`class_counts.csv` | 可以导入笔记或表格，复核数字 |

各数据集生成三张图，按相同类型并排阅读即可；若导师要求总计三张，可以后续各拼成一张双面板对照图，不需要重算统计。

已生成本地VisDrone训练集统计，见[summary.json](../../outputs/dataset_statistics/visdrone_train_v1/summary.json)：6471张图片、343204个有效框，每图平均53.04个目标，中位42个，最多902个。AI-TOD真实train/val统计也已完成，分别为11214/2804张图、282580/70424框。三类训练集对照图已查看；描述性统计和头部读取不等于完整审计或重新核验历史服务器训练数据。

统计口径统一采用bbox的原图w×h，不使用可能表示分割面积的COCO `area`字段。COCO类别通过`categories`映射，不假定`category_id-1`就是连续类别号。忽略ignore/crowd、VisDrone score!=1及非有效类别；非正/非有限框记录后跳过；越界框记录并保持原尺寸，不静默裁剪。这是描述性统计规则，不自动等于训练清洗或官方评测协议。

## 4. 补齐图片与核对标注

若实验室已有合法获取的原版完整副本，可直接核对来源、标注版本和划分，避免重复生成。否则按官方README：下载xView训练图像/geojson与AI-TOD非xView图片部分，clone仓库后按`aitodtoolkit`要求放置文件，在该目录运行`python generate_aitod_imgs.py`。

作者文档列出的生成环境为Python3.7、mmcv、wwtool等旧工具；应放在独立生成环境，保留当前YOLO环境。README未给出完整依赖锁，旧脚本含`np.float`等旧API，不能承诺在当前Python3.13/新版numpy直接运行。当前三类统计完全不依赖这些旧工具，因此不必等生成环境准备好才开始。

图片齐全后，可以在新输出目录增加`--image-dir`，其根目录应让`根目录 / JSON中file_name`确实指向图片。例如`file_name`仅为`xxx.png`时，根目录选`AI-TOD/images/train`；如果已经含`train/xxx.png`，就选`AI-TOD/images`。

```bash
uav-detect statistics \
  --format coco \
  --annotations /data/zqq/AI-TOD/annotations/实际train文件名.json \
  --image-dir /data/zqq/AI-TOD/images/train \
  --name AI-TOD-original --split train \
  --output-dir outputs/dataset_statistics/aitod_train_images_v1
```

该选项检查文件存在/图片头部尺寸，失败会写报告并以退出码2提示；不等于完整像素解码或随机框可视化。后续训练前仍需完成坏图检查、随机原图标注叠加、划分泄漏检查与标签转换。当前不需要先把AI-TOD转成YOLO格式才能统计，VisDrone转换脚本也不能直接套到COCO JSON上。

## 5. 怎样算本轮统计验收完成

- 两数据集的train各有摘要和三类图；val单独记录，不把两个split拼成一个数字。
- JSON关联关系正确，图片/类别/标注id不重复，异常有记录；只有标注时明确写「未完成图片检查」。
- 能指出目标尺寸、每图密度和类别分布的差异，解释不能互替：视角/来源、类别体系、标注和尺寸分布不同；均为极小目标场景不代表相同分布。
- 论文写全量28,036图、700,621实例、平均尺寸约12.8像素，单个train split无需硬凑这些总量。实际版本、筛选和划分必须写清。
- 三类分布只能说明数据特点，不计算AP，不证明跨数据集泛化或科研创新。后续AI-TOD的AP判据应使用约定的官方评测，不能直接套本Demo的small<32协议。

完成后提交两个`summary.json`及对应三类图，先独立说出至少两点差异，再继续训练管线。AI-TOD图片尚未齐全时可以先验收「标注统计」，不将其登记为完整数据管线完成。
