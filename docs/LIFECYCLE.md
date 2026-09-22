# 闭环规则

## 初始化（每类一次，`prepare_simulation_streams.py`）

数据集 `test` 目录为固定测试集，只用于评测，不流入、不训练、不标定。`train` + `val` 按 SHA256 去重后：seed 42 固定抽取：32 张 OK 作残差参考图，200 张 OK 作 PatchCore 记忆库（同时永久锁定为 YOLO 训练 OK），
200 张 OK 作校准集（永远不进任何训练）。这 432 张不进入产线流。斜拍 OK 少，改为 32 / 100 校准 / 100 记忆库（`--reserve qiumian_xiepai=32,100,100`）。
其余图片默认全部流入（NG 比例可选，默认全部 NG），OK 与 NG 按比例均匀分到每个批次（`--stream-mode stratified`），每 200 张一个复核批次，斜拍每 100 张（`review_batch_size`）。

## 预训练检测器（`plugins/pretrained_patchcore.py`）

- 特征：DINOv2-large 四层 patch 特征，与 32 张参考图逐 patch 余弦最近邻相减得残差，经 ADPretrain `pro_angle` 投影器。
- 记忆库：200 张 OK 的投影残差 patch，coreset 2%，最近邻用 torch 精确搜索（无 faiss 依赖）。记忆库缓存到 `pretrained_cache/<category>/memory_bank.npy`。
- 热力图：patch 到记忆库的最近邻距离，上采样加高斯平滑；外圆 ROI 外置最低值。
- 图像得分：ROI 内最高 0.5% 像素均值。得分 ≥ 图像阈值判 NG。
- 分割（只对 NG）：像素阈值 = max(校准 OK 像素 q99.7, 0.7 × 本图峰值)，8 连通去掉 < 128 像素，按峰值取 Top-3，mask 为并集。
- 得分在阈值 ±5% 内标为边界样本。

## 阈值规则（`calibration.py`，预训练与 YOLO 共用）

| 确认 NG 数 | 规则 |
|---|---|
| 0 | 校准 OK 得分分位：外圆 q95，其余 q90 |
| 1 到 29，每 5 张重算 | OK 分位阶梯 q80/q90/q95/q97/q99/q99.5/q99.9 里能抓住全部已知 NG 的最高档，都抓不住退到 q80 |
| ≥ 30 | 召回优先：误报 ≤ 0.2 的阈值里，召回 ≥ 目标（外圆 0.99，其余 0.95）的选误报最低；目标不可达时取上限内 Youden（召回 − 误报）最优点，而不是上限边缘 |

YOLO 的得分 0（无检出）永远不是合法阈值。阈值文件附带召回阶梯（1.0/0.95/0.9/0.85/0.8 对应阈值与误报）。

## YOLO-seg 里程碑（`training/seg_lifecycle.py`）

- 触发：有效 NG（确认且 mask 合格）达到 40，之后每 +20，100 张后每 +40；外圆从 40 起每 +40（`retrain_increment` 按类别配置）。每个里程碑只用最早的 N 张 NG。
- 划分：NG 按 (seed, 类别, sha256) 哈希一次性分到训练 75% / 校准 25%，永不改动。校准 OK = 初始化保留的 200 张。
- 训练 OK：记忆库 OK + 流内确认 OK，按批次分层轮流抽样，最多 400 张。
- 数据集：mask 转多边形 label，一个连通域一个实例；外圆 ROI 外填白、真值与 ROI 求交，缺陷完全在 ROI 外的 NG 剔除并记录。
- 训练：yolo26s-seg，1024，batch 4，100 epoch 固定取 last.pt，mosaic 1.0，copy_paste 0.3。校准集兼作 Ultralytics 验证集（不做模型选择，无泄漏）。
- 标定：图像阈值按上表的召回优先规则；mask 置信度阈值取校准 NG 平均 IoU 最高的一档。
- 推理：conf 下限 0.001，图像得分 = 最高实例置信度；NG 图输出置信度 ≥ mask 阈值的实例并集、框和逐像素置信度图。

## 复核抽检（`review/sampling.py`）

每批推理结束后模拟人工复核（仿真按数据集路径和 mask）：模型判 NG 的全部复核；模型判 OK 的按得分从高到低分三段，
最高 20% 全检，中间 40% 抽 10%（至少 5 张），最低 40% 抽 2%（至少 2 张）。某段抽检发现 NG 则该段转为全检。
未复核的样本记为伪 OK（`sampling_pseudo_ok_pool`），只作为 YOLO 训练 OK，不进校准。仿真额外记录伪 OK 里实际藏有的 NG 数
（`review_sampling.pseudo_ok_hidden_ng`），只用于评估抽检策略，不参与任何决策。参数在 `configs/pipeline.yaml` 的 `review_sampling`。

## 离线门槛、影子运行与切换

1. **离线门槛**：候选训练完立即在校准集上与当前正式模型比较（正式模型现场重新打分）：漏检不增、误报不升（`max_fpr_increase` 默认 0）、
   有实际收益（漏检更少或误报至少降 0.01）。不通过 → `rejected_offline`，不进影子。
2. **影子运行**：通过的候选从下一批起与正式模型并行推理，只累计已复核样本的判定。
3. **判定**：累计复核 NG ≥ `shadow_min_ng`（10）或满 `shadow_max_batches`（5）批时，用同样三条规则比较两者在影子样本上的漏检与误报：
   通过 → `registry.promote`，候选成为正式模型；否则 `rejected_shadow`。影子期间若到达新里程碑，旧候选标 `superseded`，新候选重新走门槛与影子。
4. 切换后不保留旧模型影子。结果写 `promotion_reports/<category>/offline_<模型>.json` 与 `shadow_<模型>.json`。

## 固定测试集评测（`evaluation.py`）

每个进入产线的模型都在固定测试集上评测并记录：预训练在每次阈值重标定后（角色 production），YOLO 候选训练完成后（角色 candidate），切换为正式模型时再记一次（角色 production）。
指标：召回、OK 误报率、精确率、AUROC、NG 前景 micro IoU（主指标），保留逐图平均 IoU 作为辅助值。
只检测 ROI 内缺陷，漏检图的 GT 像素仍计入 micro IoU 并集。缺失/无效 GT 会使分割指标不可用，不当成 0 分。
在线批次与汇总仅统计已验证的审核子集，明确排除初始化 OK 和伪标签，不代表完整在线流。
逐图得分、预测 mask、热力图、带框图缓存在 `workspace/test_cache/<category>/<模型>/`；数据、GT、ROI、权重或推理参数变更会使缓存失效，
只改图像分类阈值时仍可复用原始预测。具体口径、字段和兼容说明见 [METRICS.md](METRICS.md)。

每次评测生成 `workspace/test_reports/<category>/<时间>_<角色>_<模型>/`，按 tp/fp/fn/tn 分目录，每张图一个文件夹：
`original.<ext>`（原图硬链接）、`original_mask.<ext>`（NG 的原始 GT，硬链接）、`pred_mask.png`、`heatmap.jpg`（预训练：热力图；YOLO：逐像素最高实例置信度）、
`boxed.jpg`（预训练：叠加热力图与 Top-3 区域框及峰值；YOLO：保留实例框及置信度）、`score.json`（得分、阈值、判定、IoU、区域或实例列表）。
另有 `cases.csv` 与 `report.json`，`test_history.jsonl` 汇总全部评测。可视化可用 `fixed_test_visuals: false` 关闭。

## 输出

```text
results/<RunName>/workspace/state/lifecycle_<category>.json      批次、里程碑、事件历史
results/<RunName>/workspace/batch_reports/<category>/            每批分类与分割指标、影子不一致清单
results/<RunName>/workspace/promotion_reports/<category>/        每个里程碑的离线比较
results/<RunName>/workspace/test_reports/<category>/             每个产线模型的固定测试集评测（test_history.jsonl 汇总）
results/<RunName>/workspace/model_registry/<category>/           milestones/（训练与标定明细）、versions/、production.json
results/<RunName>/pretrained_cache/<category>/                   记忆库、校准 OK 得分、thresholds.json（含历史）
results/<RunName>/pretrained_artifacts/<category>/<sample>/      热力图、mask、regions.json、overlay.png
```
