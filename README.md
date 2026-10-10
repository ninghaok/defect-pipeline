# pipeline — 冷启动预训练检测 → YOLO 实例分割闭环

四个工位（球面俯拍、球面斜拍、球塞底面、球塞外圆）的异常检测与缺陷分割闭环。每个类别从零 NG 开始：

1. **预训练检测器**（ADPretrain 残差特征 + PatchCore，DINOv2-large，224 单视图）承担冷启动阶段的分类，
   图像得分 = 热力图最高 0.5% 像素均值；判 NG 的图用**混合规则**（OK 像素 q99.7 与 0.7×本图峰值取大）切出 Top-3 区域。
2. 复核确认的 NG 每累计 5 张重新标定预训练阈值（零 NG：OK 分位；< 30 张：OK 分位阶梯；≥ 30 张：召回优先 + 误报上限）。
3. 每批复核：模型 NG 全检，模型 OK 按得分分段抽检（高 20% 全检，中 40% 抽 10%，低 40% 抽 2%，抽到 NG 即该段全检），未检样本作伪 OK 只进训练。
4. 有效 NG 达到 40 张训练第一个 **YOLO26s-seg** 候选（检测置信度判 NG，NG 图输出实例 mask 并集），
   完成阈值校准和推理冒烟后直接影子运行，累计 10 张有效复核 NG 或满 5 批后判定。
   只在影子样本上比较：C=2FN+FP 降低，或 C 持平且 T₅ 提高，或 T₅ 提高至少 20 个百分点，即可切换。
   校准集只确定阈值，不再与正式模型比较并提前否决候选。
5. 之后每 20 张 NG（100 张后每 40 张；外圆每 40 张）重训一次，同样的影子评估。每个模型都在固定测试集上评测并记录。

当前实现的基准说明见 [docs/BASELINE.md](docs/BASELINE.md)；以前 PR 中与它不同的描述已被替代。

详细规则见 [docs/LIFECYCLE.md](docs/LIFECYCLE.md)，环境与运行见 [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)。

## 目录

```text
cli/run_lifecycle.py              单类别闭环（预训练 → 里程碑训练与阈值校准 → 影子评估 → 切换）
cli/prepare_simulation_streams.py 从 dataset_523 生成仿真流：test 为固定测试集；train+val 中 32 参考 OK + 200 记忆库 OK + 200 校准 OK（斜拍 32+100+100）永久保留，其余按 NG 比例流入
cli/yolo_seg_ng_count_ablation.py NG 数量消融实验（固定划分，test 只报告）
cli/audit.py / report_metrics.py  工作区完整性审计、分类与分割指标汇总
scripts/run_lifecycle_windows.ps1 四类闭环一键运行
scripts/run_seg_ng_ablation_live.ps1  消融实验运行
src/detected_pipeline/calibration.py           三段式阈值规则（OK 分位 / 分位阶梯 / 召回优先），共用
src/detected_pipeline/pretrained/              特征网络、PatchCore 记忆库、混合分割
src/detected_pipeline/plugins/pretrained_patchcore.py   预训练插件（分类 + 分割 + 阈值重标定）
src/detected_pipeline/plugins/yolo_supervised.py        YOLO-seg 插件（检测优先，NG 才分割）
src/detected_pipeline/training/runner.py       yolo26s-seg 固定轮数训练、冒烟
src/detected_pipeline/training/seg_lifecycle.py NG 持久划分、训练 OK 分层抽样、数据集生成、阈值标定
src/detected_pipeline/feedback|registry|review 回流池（SQLite）、模型注册与切换、文件夹真值审核（仿真）
models/   yolo26s-seg.pt, dinov2_vitl14_pretrain.pth, checkpoints_pro_angle.pth（只放用到的三个）
vendor/adpretrain/   ADPretrain 的 DINOv2 封装、角度投影器、PatchCore（本地权重，无网络下载）
roi/qiusaiwaiyuan.png   外圆 ROI（白色为检测区）
```

## 快速运行

```powershell
conda activate pipeline
# 在克隆后的仓库根目录执行
python .\cli\check_local_installation.py
# 先按 docs/SYNTHETIC_AUGMENTATION.md 配置生成后端
$env:PIPELINE_SYNTHETIC_CONFIG = "C:/path/to/local-synthetic.yaml"
.\scripts\run_lifecycle_windows.ps1 -RunName lifecycle_seg_01
```

结果位于 `results\<RunName>`：`workspace\batch_reports\<category>` 每批指标，`workspace\promotion_reports` 每个候选的影子比较，
`workspace\test_reports\<category>` 预训练模型和每个 YOLO 版本的固定测试集评测（按 tp/fp/fn/tn 保存原图、原始 mask、预测 mask、热力图、带框图），`workspace\model_registry\<category>` 候选与生产模型，`pretrained_cache\<category>\thresholds.json` 预训练阈值历史，
`reports\classification_metrics.*` 汇总。

## 边界

- 推理阶段不读取 OK/NG 路径和 mask；仿真审核器在推理之后才读取隐藏真值。真实产线需替换为人工审核服务。
- 校准 OK（200 张）永远不进 YOLO 训练；NG 一经确认即按哈希永久分到训练或校准。
- 训练固定 100 epoch 不早停，发布 `last.pt`；数据集 test 目录是固定测试集，预训练模型和每个训练完成的候选都在其上评测，从不参与训练、标定或选择。
- 四类的数据、阈值、模型完全隔离。
# 默认启用的伪缺陷训练增强

默认启用的限额增强已接入候选模型训练，支持独立生成环境、比例和次数上限、面积匹配、重启审计及逐图溯源。配置、隔离规则和对照实验说明见 [SYNTHETIC_AUGMENTATION.md](docs/SYNTHETIC_AUGMENTATION.md)。
