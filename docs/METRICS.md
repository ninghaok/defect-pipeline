# 评测指标约定

本约定适用于在线批次、在线汇总、固定测试集和单独 YOLO 测试。报告中的
`primary_segmentation_metric` 为 `iou_micro`。标定阈值的优化目标和模型晋升规则不在本次修改范围内。

## 真值与样本范围

- 在线 `official` 和最终汇总只使用 `human_review`、`folder_ground_truth` 来源的已审核在线样本。
  `scope=reviewed_online_subset` 表示审核子集；抽检具有选择偏差，不能当成完整在线流表现。
- 初始化建库 OK、`sampling_pseudo_ok`、`model_assumed_review` 和未审核记录不能充当评测真值。
  汇总披露初始化、未验证和缺失推理记录的数量。
- 仿真 `hidden_truth` 仍只用于抽检诊断，不进入评测用的审核真值，也不回流训练或晋升。
- 独立测试集单独报告，不与在线样本合并。没有真实 NG 或 OK 时，对应 Recall 或误报率为 `null`。
- `cli/export_simulation_table.py` 在仿真结束后独立读取整批文件夹真值，输出
  `scope=complete_simulated_batch` 的完整在线流指标。它只写汇总文件，不向训练、校准或晋升回传真值；
  不能将它与上述已审核子集指标混称为同一统计范围。

## ROI 与缺失标注

ROI 白色区域为检测域。GT 和预测 mask 都与相同 ROI 求交；YOLO 保留实例的 mask 也裁到 ROI 内。
存在缺陷但完全位于 ROI 外的 NG 延续原固定测试集规则：排除并计数，不改标成 OK。
这条规则只统一评测域；不改训练池、抽检选择和晋升判定的标签策略。

外部数据集 mask：黑色缺陷；人工审核内部 mask：白色缺陷。NG mask 必须可读、与图像尺寸一致，且原始 mask 存在缺陷像素。

固定测试集、在线报告对缺失、不可读、空缺陷或尺寸不符的 GT 标记具体 `gt_status`。
存在任何无效 NG 标注时，本次分割主值与辅助均值均为 `null`，同时报告
`segmentation_valid_ng`、`segmentation_invalid_ng` 和 `segmentation_status`，不会偷偷用有效子集发布完整指标。
没有 ROI 时，已知的图像级 NG 标签仍可用于分类；有 ROI 时无法确认缺陷是否在检测域内，因此排除该图的分类统计并计入 `excluded_invalid_gt`。
单独 YOLO/消融的数据加载入口在缺少必要 mask 时会明确报错，不输出伪造的零分。

## 分割主指标

主指标只针对**有效真实 NG 图像**的缺陷前景，OK 图像的误报通过 OK 误报率单独汇报。
它不是对所有图像、所有语义类别求平均的 mIoU。

对于每张 NG，将 GT 和预测限制在 ROI 内。最终分类为 OK 时，预测 mask 视为空。

`iou_micro = 所有 NG 的交集像素数之和 / 所有 NG 的并集像素数之和`

漏检图的交集为 0，但其 GT 面积仍进入并集，不能简单删除该图，也不能按检出的 NG 单独汇总。
例如：1 像素缺陷完全检出，100 像素缺陷漏检，micro IoU 为 `1/101`。

`mean_iou_all_ng` 保留为辅助的逐图算术平均，例子中是 `0.5`，不改名冒充 micro IoU。
`dice_micro` 同样先汇总像素再计算。旧在线 `iou`、`dice` 字段分别为 micro 指标的兼容别名。
汇总 `iou_mean` 保留逐图均值语义；`false_positive_rate` 为 `ok_false_positive_rate` 的兼容别名。
旧 `dice_mean` 不再输出，使用明确命名的 `dice_micro`。

固定测试的 `cases.csv` 和 `score.json` 同时提供：

- `raw_localization_iou`：图像分类门控前的原始定位 IoU；旧 `iou` 保留这一含义。
- `end_to_end_iou`：最终分类门控后的逐图 IoU。
- `intersection`、`union`：最终门控后的像素计数，可相加后复算 micro IoU。
- `gt_status`：标注有效性。缺失项留空，不能补零。

固定测试的预测图保留原始定位输出，用于分析“有 mask 但分类漏检”的现象；它不等于最终产线 mask。
GT 问题与 ROI 排除明细写入 `report.json` 的 `gt_issues`。

## AUROC 和缓存

主线与消融共用 AUROC 实现：NG 分数高于 OK 计 1，相等计 0.5，低于计 0；缺少任一类时为 `null`。

固定测试缓存验证有序样本的路径、图像内容、标签、GT 内容、ROI 内容、模型键和推理参数。
主入口模型键覆盖 YOLO 权重、mask 阈值和推理参数；预训练覆盖记忆库、分割参数和像素阈值。
单独改变图像分类阈值不会改变原始 score/mask，可以复用推理，再计算最终指标。
外部直接调用 `score_fixed_test` 时，应提供能标识真实模型的 `model_key`，并通过 `inference_settings` 描述影响原始预测的参数。

预训练记忆库的指纹包含编码器/投影器权重和 backbone；校准缓存还核验得分参数、分割参数、阈值规则及校准图内容。
原有阈值失效时，用原有确认 NG 重新评分，不把确认 NG 数归零。

旧缓存缺少新身份信息时自动失效。新的推理身份使用独立可视化目录，避免修改旧报告硬链接的预测图。
首次运行可能重建记忆库或重新推理；不会自动把旧版报告改写为新口径。历史实验应重新评测后再比较。
