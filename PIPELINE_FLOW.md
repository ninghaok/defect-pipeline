# 完整流程描述（含切换规则、模型策略与参数）

代码对应：`cli/run_lifecycle.py`（主循环）、`plugins/pretrained_patchcore.py`（预训练）、`plugins/yolo_supervised.py`（YOLO-seg）、
`training/seg_lifecycle.py`（里程碑训练与门控）、`calibration.py`（阈值规则）。参数来自 `configs/pipeline.yaml`、`configs/pretrained.yaml`、`configs/training.yaml`。

## 0. 初始化（每类一次）

`cli/prepare_simulation_streams.py`：数据集 `test` 目录为固定测试集（只评测，不流入、不训练、不标定）；`train` + `val` 去重后 seed 42 抽取并永久保留：

| 保留集 | 数量 | 用途 |
|---|---:|---|
| 参考 OK | 32 | 预训练残差参考图（实际由 k-center 在参考+记忆库 OK 中重选 32 张） |
| 记忆库 OK | 200，斜拍 100（类别 OK 不足时自动缩减，至少留 100 张进流） | PatchCore 记忆库；同时永久锁定为 YOLO 训练 OK |
| 校准 OK | 200，斜拍 100 | 预训练与 YOLO 的校准 OK，永远不进任何训练 |

斜拍 OK 总数少，保留改为 32 参考 / 100 校准 / 100 记忆库（共 232 张）。其余图片默认全部流入（NG 比例可选，默认全部 NG），OK 与 NG 按比例均匀分到每个批次，
每 200 张一个复核批次，斜拍每 100 张（`review_batch_size` 按类别配置）。

## 1. 预训练检测器（0 NG 起的正式模型）

**模型**：DINOv2-large（本地权重 `models/dinov2_vitl14_pretrain.pth`）第 6/12/18/24 层 patch 特征；每个 patch 与 32 张参考图的 patch 做余弦最近邻并相减得残差；残差经 ADPretrain 角度投影器 `models/checkpoints_pro_angle.pth`；PatchCore 记忆库 = 记忆库 OK 的投影残差 patch，近似贪心 coreset 2%，最近邻用 torch 精确搜索。记忆库缓存于 `pretrained_cache/<category>/memory_bank.npy`，重启只重编码 32 张参考图。

**输入**：整图 letterbox 到 224；外圆先按 `roi/qiusaiwaiyuan.png` 裁到外接框、ROI 外填均值色。

**分类**：热力图 = patch 到记忆库最近邻距离，双线性上采样到裁剪尺寸并高斯平滑（σ=4），ROI 外置最低值。图像得分 = ROI 内最高 0.5% 像素均值。得分 ≥ 图像阈值 → NG；得分在阈值 ±5% 内标记为边界样本。

**分割（只对 NG）**：像素阈值 = max(校准 OK 像素 q99.7，0.7 × 本图峰值)；8 连通，去掉 < 128 像素，按峰值排序保留前 3 个区域；mask = 保留区域并集；输出每个区域的框、峰值、均值、面积和叠加图。

**参数**（`pretrained.yaml`）：image_size 224、reference_size 32、coreset 0.02、top_fraction 0.005、pixel_quantile 0.997、peak_fraction 0.7、min_area 128、max_regions 3、boundary_ratio 0.05。单张约 45 ms。

## 2. 预训练阈值随 NG 积累重标定

每批复核后，确认 NG 数每跨过 5 的倍数（`pretrained_recalibrate_every_ng`）重算一次，用全部确认 NG 的预训练得分：

| 确认 NG | 规则 | 说明 |
|---|---|---|
| 0 | 校准 OK 得分分位 | 外圆 q95，其余 q90 |
| 1 到 29 | OK 分位阶梯 q80/q90/q95/q97/q99/q99.5/q99.9 | 取能抓住全部已知 NG 的最高档；都抓不住退到 q80（= 误报上限 0.2） |
| ≥ 30 | 召回优先 | 误报 ≤ 0.2 的候选中召回 ≥ 目标（外圆 0.99，其余 0.95）的选误报最低；不可达时取上限内 Youden（召回 − 误报）最优点 |

阈值与历史写入 `pretrained_cache/<category>/thresholds.json`，附召回阶梯（1.0/0.95/0.9/0.85/0.8 各自的阈值与误报）。预训练切为影子后仍继续重标定，影子结束即停止。

## 3. 复核与回流

推理只读图像。每批推理结束后模拟人工复核（仿真按 OK/NG 目录与 mask）：模型判 NG 的全部复核；模型判 OK 的按得分分段抽检——最高 20% 全检，
中间 40% 抽 10%（≥ 5 张），最低 40% 抽 2%（≥ 2 张），某段抽到 NG 即该段全检。未复核样本记为伪 OK，只作 YOLO 训练 OK。
确认 OK 进 confirmed_ok_pool；确认 NG 且 mask 合格的进 labeled_pool（计入里程碑）；模型 NG 而真值 OK 进历史误报池，模型 OK 而真值 NG 进历史漏检池。
每批写 `batch_reports/<category>/batch_xxxx.json`（已复核样本的分类、分割指标，抽检统计与伪 OK 中隐藏 NG 数，影子分歧）。

## 4. YOLO 里程碑训练

**触发**（`next_milestone`）：有效 NG ≥ 40 训练 v40；之后每 +20（60、80、100）；100 之后每 +40（140、180…）；外圆从 40 起每 +40（80、120…）。每个里程碑只用最早确认的 N 张 NG，同一里程碑只训练一次。

**数据**：
- NG 按 (split_seed 42, 类别, sha256) 哈希一次性分到训练 75% / 校准 25%，记录在 SQLite，永不改动。
- 训练 OK = 记忆库 OK + 流内确认 OK，排除校准 OK，按批次分层轮流抽样，最多 400 张（`train_ok_limit`），少于 100 张不训练。
- mask → 多边形 label，每个连通域一个实例，单类 `defect`；外圆 ROI 外填白、真值与 ROI 求交，缺陷完全在 ROI 外的 NG 剔除并记录。
- 校准集 = 200 校准 OK + 校准 NG，兼作 Ultralytics 验证集（固定轮数，不据此选模型）。

**训练**（`training.yaml`）：基座 `models/yolo26s-seg.pt`，task segment，imgsz 1024，batch 4，epochs 100 固定不早停取 last.pt，seed 42，AMP，deterministic，mosaic 1.0（最后 10 轮关闭），copy_paste 0.3，overlap_mask，mask_ratio 4。

**标定**：候选在校准集上推理（conf 下限 0.001，NMS IoU 0.7，max_det 100，retina_masks），图像得分 = 最高实例置信度：
- 图像阈值：召回优先（目标 0.95，外圆 0.99，误报上限 0.2，不可达用 Youden 回退），得分 0 永不作为阈值。
- mask 置信度阈值：在 0.001 到 0.999 网格上取校准 NG 平均 IoU 最高的一档。

候选经冒烟（加载 + 单张推理）后登记到 `model_registry/<category>/versions/<model_version>/`。

## 5. 离线门槛、影子运行与切换

1. **离线门槛**：候选训练完在校准集上与当前正式模型比较（正式模型现场重新打分，各用自己的阈值）：候选漏检 ≤ 正式；候选误报率 ≤ 正式 + `max_fpr_increase`（0）；
   且漏检更少或误报至少降 `min_fpr_reduction_for_equal_fn`（0.01）。不通过 → `rejected_offline`。
2. **影子运行**：通过的候选从下一批起并行推理，只累计已复核样本。
3. **判定**：累计复核 NG ≥ `shadow_min_ng`（10）或满 `shadow_max_batches`（5）批，用同样三条规则比较影子样本上的漏检与误报：通过 → 切换为正式模型；否则 `rejected_shadow`。
   影子期间到达新里程碑则旧候选 `superseded`。切换后不保留旧模型影子。

**固定测试集评测**：每个进入产线的模型（预训练每次重标定后、每个 YOLO 候选、切换时）在固定测试集上评测召回、误报、AUROC、NG IoU，写入 `test_reports/<category>/<时间>_<角色>_<模型>/`，
按 tp/fp/fn/tn 分目录保存原图、原始 mask、预测 mask、热力图、带框图和 score.json；逐图结果缓存，每个模型只推理一次。

## 6. YOLO-seg 推理策略（切换后的正式模型）

1. 外圆按 ROI 填白后送入；`predict(conf=0.001, iou=0.7, max_det=100, retina_masks=True, imgsz=1024)`，得到全部低置信候选实例；外圆丢弃与 ROI 不相交的实例。
2. 图像得分 = 最高实例置信度，无检出为 0。得分 ≥ 图像阈值 → NG；得分在阈值 ±5% 内标边界。
3. 判 NG 的图：输出置信度 ≥ mask 阈值的实例并集 mask、各实例框与置信度、逐像素最高置信度图、叠加图；判 OK 的图 mask 为空。
4. 单张 12 到 20 ms（RTX 5080）。

## 7. 单独训练

`cli/train_yolo.py`（`scripts/train_yolo_windows.ps1`）用第 4 节相同的方法对 train/val/test 目录训练、标定并评测，输出与注册表同格式的 `summary.json`，可直接交给 `YoloFeedbackAdapter` 推理。见 `docs/TRAIN_YOLO.md`。

## 8. 关键参数一览

| 参数 | 值 | 位置 |
|---|---|---|
| review_batch_size | 200（斜拍 100） | pipeline.yaml |
| ng_calibration_fraction / split_seed | 0.25 / 42 | pipeline.yaml lifecycle |
| train_ok_limit / min_train_ok | 400 / 100 | pipeline.yaml lifecycle |
| pretrained_recalibrate_every_ng | 5 | pipeline.yaml lifecycle |
| first_train_ng / retrain_increment / after / late | 40 / 20（外圆 40）/ 100 / 40 | pipeline.yaml lifecycle |
| shadow_min_ng / shadow_max_batches | 10 / 5 | pipeline.yaml lifecycle |
| fixed_test_visuals | true | pipeline.yaml |
| review_sampling high/middle 比例，middle/low 抽检率与最小张数 | 0.20 / 0.40，10%（≥5）/ 2%（≥2） | pipeline.yaml review_sampling |
| pseudo_ok_in_training | true | pipeline.yaml lifecycle |
| yolo target_recall / max_fpr | 0.95（外圆 0.99）/ 0.2 | pipeline.yaml lifecycle.yolo_thresholds |
| promotion max_fpr_increase / min_fpr_reduction | 0.0 / 0.01 | pipeline.yaml lifecycle.promotion |
| 预训练 zero_ng_quantile | 0.90（外圆 0.95） | pretrained.yaml thresholds |
| 预训练 ladder_quantiles / full_rule_min_ng | q80…q99.9 / 30 | pretrained.yaml thresholds |
| 预训练 image_size / reference_size / coreset / top_fraction | 224 / 32 / 0.02 / 0.005 | pretrained.yaml |
| 预训练分割 pixel_quantile / peak_fraction / min_area / max_regions | 0.997 / 0.7 / 128 / 3 | pretrained.yaml segmentation |
| YOLO epochs / imgsz / batch / mosaic / copy_paste | 100 / 1024 / 4 / 1.0 / 0.3 | training.yaml |
| YOLO conf_floor / nms_iou / max_det | 0.001 / 0.7 / 100 | training.yaml inference |
