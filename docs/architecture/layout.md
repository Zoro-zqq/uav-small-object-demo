# 目录与扩展约定

## 文件放置

| 内容 | 位置 | 规则 |
|---|---|---|
| 原图、原标注、转换数据 | 项目外数据目录 | 原始数据只读；转换版本另建，协议写入报告 |
| 数据集路径 | `configs/datasets/*.local.yaml` | 本机文件默认忽略；示例使用`.example.yaml` |
| 实验配方 | `configs/experiments/<dataset>/<model>.yaml` | 记录模型、数据配置、预算、种子和输出名 |
| 通用配置/路径 | `src/uav_detection/common/` | 不导入模型框架或数据集私有类别 |
| 数据集字段与类别 | `datasets/<dataset>.py` | 不复用VisDrone类别假装支持AI-TOD |
| 格式转换 | `datasets/converters/<source>_to_<target>.py` | 写清类别映射、坐标、无效框和忽略策略 |
| 模型框架 | `backends/` | 延迟导入GPU依赖，新增框架需实现适配 |
| 外部论文代码配置 | `experiments/` | 保持外部仓库独立，不复制整个mmdet到本项目 |
| 数值与图表 | `outputs/<类型>/<唯一实验名>/` | 同条件同协议比较；不要挪动或修改历史结果 |
| 实验解释与结论 | `docs/reports/` | 链接产物，注明协议、设置和局限 |
| 学习过程 | `docs/learning/` | 保留历史；论文阅读与跨文知识统一在Obsidian |

## 新增一个模型基线

已有Ultralytics支持的模型只需新增实验YAML，设置`backend: ultralytics`与实际模型权重/YAML。训练入口复用，不复制一份train脚本。切换数据集需先提供正确类别和对应转换标签。当前尺寸分组评价依赖VisDrone十类，扩展到AI-TOD前必须把类别参数化并验证协议，不能仅改数据路径。

新的模型框架在`backends/`提供适配器并注册；训练、验证返回值和元数据要明确。当前工厂仅加载Ultralytics模型，不能认为所有框架已经共享训练/评价API。RemDet通过`prepare-remdet`生成短跑配置，`prepare-remdet-eval`生成纯评估配置；实际运行仍使用其官方`tools/train.py`或`tools/test.py`。`summarize-remdet`只整理完成的测试指标，不用预测pkl冒充指标JSON。

## 新增数据集或转换器

先建立数据与标注协议，再写转换器。共用图像读取和路径工具，数据集的类别、坐标及ignore语义留在自己的模块中。生成可审计统计，默认拒绝覆盖已有产物。加入一个小合成样本检查非连续category id、边界框、空图及ignore情况，再用少量真实样本叠框核查。

本次整合的VisDrone→YOLO保留原有裁边逻辑，VisDrone→COCO则保留原始xywh、独立记录排除条目。两者是不同转换协议，不能把数量或AP当作相同输入直接比较。

## 路径与环境

在仓库内的editable安装默认以仓库根目录解析配置/产物路径；可用`UAV_PROJECT_ROOT`显式指定工作区。在普通wheel安装后默认以当前调用目录作为工作区。数据YAML里的相对路径相对其`path`字段或YAML所在目录解析。

核心依赖只用于数据与分析；Ultralytics、MMEngine分别按用途安装。各框架的PyTorch/MMCV/CUDA组合在自己的环境管理，不将所有训练框架强装进同一环境。

## 历史与版本控制

本次不自动提交、推送或改动git暂存区。已有未提交修改随文件迁移保留，用户已删除的`conclusion.md`不恢复。既有outputs不移动、不重写；`.gitignore`只限制未来新增产物，已跟踪历史仍受git管理。
