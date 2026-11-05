# 本地验证记录

日期：2026-11-05。源码参照本机RemDet commit `8cf2667e9ff80122e89436a0ebc6558ff1cfdb93`。

已通过：

- 两个Python脚本的语法检查。
- 合成train/val图片及8列原生标注的核心转换：类别1–10映射、类别0/11及score0排除、无效尺寸与图外框排除、遮挡框保留、跨边界xywh保持不变、排除条目统计。
- 非预期类别、错误列数、图片与TXT不匹配时明确报错。
- 使用MMEngine0.10.4真实加载官方配置，生成两套短跑配置并重新读取；核对模型类型、宽度、batch与优化器字段、lr、训练/验证/测试路径、评价标注路径和日程。
- 配置生成器拒绝覆盖已有目录。

限制：本机Windows未授予创建软链接权限，转换命令在创建链接处遇到WinError1314，因此未声称整个CLI及软链接流程通过；配置检查使用独立合成COCO目录，不依赖这次失败输出。目标为用户Linux服务器，真实目录和Linux软链接仍待执行检查。没有检查真实VisDrone标注，没有运行GPU模型或训练，没有得到本人AP。

MMEngine及少量验证依赖安装在本机临时验证目录，不代表服务器环境已安装。

## 官方权重评估工具更新

2026-11-05用户将范围改为仅评估官方RemDet-Tiny权重，数据路径更新为`/home/lx/data/zqq/VisDroneDataset`。新增`convert-coco --split val`、`prepare-remdet-eval`及`summarize-remdet`，操作以[评估指南](../guides/remdet-official-eval.md)为准。

- 全部40项测试通过；新增用例检查仅有验证集时的转换、拒绝混合多次test结果、AP百分数单位和无训练入口的评估配置。
- MMEngine0.10.4真实读取本机官方Tiny/S/M/L/X配置，生成及回读成功；逐一断言模型完整字典、测试pipeline和proposal_nums与来源一致。
- `git diff --check`通过。MMCV官方cu121/torch2.2索引可访问，含cp311 Linux x86_64的2.2.0 wheel。
- 尚未下载或严格加载真实权重，未在Linux服务器安装环境、转换完整真实数据或执行CUDA推理。上述40项测试不涵盖GPU或官方权重的实测性能。
