# 2026-11-05目录迁移

重构前的33项测试通过。保留用户已有未提交内容、数据路径、训练超参数、历史结果与已删除文件状态；未启动训练。

## 入口迁移

先在项目根目录`pip install -e .`，需要YOLO时安装`pip install -e ".[ultralytics]"`。旧根目录脚本已移动，不保留一组同名空壳；统一命令也可以写成`python -m uav_detection <command>`。

| 旧命令 | 新命令 |
|---|---|
| `python data_audit.py` | `uav-detect audit` |
| `python convert_annotations.py` | `uav-detect convert` |
| `python dataset_statistics.py` | `uav-detect statistics` |
| `python train.py` | `uav-detect train` |
| `python evaluate.py` | `uav-detect evaluate` |
| `python predict.py` | `uav-detect predict` |
| `python analyze_small_objects.py` | `uav-detect analyze-sizes` |
| `python analyze_failures.py` | `uav-detect analyze-failures` |

新增`convert-coco`与`prepare-remdet`整合本次论文复现的辅助脚本，见[RemDet指南](../guides/remdet-server.md)。

## 文件迁移

| 原位置 | 新位置 |
|---|---|
| `README.md`完整原流程 | `docs/guides/workflow.md`；根README改为简明入口 |
| `BASELINE_REPORT.md` | `docs/reports/baseline.md` |
| `FAILURE_MEMO.md` | `docs/reports/failure-analysis.md` |
| `LEARNING_ACCEPTANCE.md` | `docs/learning/acceptance.md` |
| `DEMO_KNOWLEDGE_NOTES.md` | `docs/learning/knowledge-notes.md` |
| `A1_A3_ACCEPTANCE.md` | `docs/plans/a1-a3-acceptance.md` |
| `AI_TOD_DATA_GUIDE.md` | `docs/guides/ai-tod.md` |
| `configs/train.yaml` | `configs/experiments/visdrone/yolov8n.yaml` |
| `configs/train_yolo11.yaml` | `configs/experiments/visdrone/yolo11n.yaml` |
| `configs/dataset.yaml` | `configs/datasets/visdrone.local.yaml` |
| `configs/visdrone.yaml` | `configs/datasets/visdrone-yolo.local.yaml` |
| 根目录Python实现 | `src/uav_detection/`，按职责拆分 |

本机配置已移动，内容未重设。服务器更新代码后，也要迁移自己的数据路径文件，不能将Windows本机路径上传后直接训练。历史outputs中的args.yaml、JSON、指纹和报告保持原样，其中旧命令/旧配置路径是历史证据，不批量重写。

## 验证结果

- 重构前33项原有测试通过，迁移后全部保留通过。
- 新增异目录调用、工作区路径、统一入口派发、全部10个子命令帮助，以及COCO转换协议回归检查；共37项通过。
- 288个既有outputs文件逐项SHA256与重构前一致，未改写历史数值、图卡或日志。
- 项目README及docs中相对Markdown链接均可解析；旧根脚本与配置引用已迁移。
- `pip install -e . --no-deps --no-build-isolation`安装及`uav-detect`实际入口检查通过；迁移后的RemDet适配器已用MMEngine0.10.4加载、导出并回读两套真实来源配置。
- 未连接服务器、未运行真实训练，新增后端接口不代表所有模型框架已适配。
