# 单独训练 YOLO-seg

`cli/train_yolo.py` 用与闭环完全相同的方法训练一个类别的 yolo26s-seg 模型，不依赖回流库，适合离线单独训练或复现。

## 数据要求

```text
<dataset-root>\<类别目录>\train\{OK,NG}     训练
<dataset-root>\<类别目录>\val\{OK,NG}       校准（选图像阈值和 mask 阈值，兼作训练验证集，不做模型选择）
<dataset-root>\<类别目录>\test\{OK,NG}      只报告，可缺省
<dataset-root>\<类别目录>\mask\<stem>_t.bmp mask：黑色为缺陷、白色为背景；也接受 <stem>_mask.* 或 <stem>.*
```

类别目录名由 `configs/pipeline.yaml` 的 `category_directory_names` 决定（qiumianfupai、qiumianxiepai、qiusaidimian、qiusaiwaiyuan），也可用 `--source-dir` 指定。外圆自动使用 `roi/qiusaiwaiyuan.png`。

## 运行

```powershell
conda activate pipeline
cd E:\ninghao\pipeline
.\scripts\train_yolo_windows.ps1 -Category di_mian_detection
.\scripts\train_yolo_windows.ps1 -Category wa_yuan_detection -TrainNgLimit 60 -Epochs 100 -SkipTest
```

参数：`-TrainOkLimit`（默认 400，seed 打乱后取前 N 张）、`-TrainNgLimit`（默认全部）、`-Epochs`、`-Batch`（显存不足改 2）、`-SkipTest`、`-Output`（默认 `C:\ninghao\results\train_yolo_<类别>_<时间>`）。
也可直接调用 `python .\cli\train_yolo.py --category ... --output ...`，其余参数同名。

## 方法（与 `configs/training.yaml`、`configs/pipeline.yaml` 一致）

- 基座 `models/yolo26s-seg.pt`，1024 输入，batch 4，100 epoch 固定不早停，取 `last.pt`，mosaic 1.0，copy_paste 0.3，AMP，deterministic。
- mask 转多边形 label，一个连通域一个实例；外圆 ROI 外填白、真值与 ROI 求交，缺陷完全在 ROI 外的 NG 剔除并写入 `split.json`。
- 推理 conf 下限 0.001，图像得分 = 最高实例置信度。
- 图像阈值：校准集上召回优先，目标召回 0.95（外圆 0.99），误报上限 0.2，目标不可达时取上限内 Youden 最优点。
- mask 置信度阈值：校准 NG 平均 IoU 最高的一档。

## 输出

```text
<output>\model\train\weights\last.pt   模型
<output>\summary.json                  checkpoint、sha256、thresholds、训练与推理参数、校准与测试指标（闭环注册表同格式）
<output>\calibration.json              阈值搜索明细、召回阶梯、逐图校准得分
<output>\test_report.json              测试指标与逐图结果
<output>\test\{tp,fp,fn,tn}\<n>\       boxed.jpg、pred_mask.png、原图
<output>\split.json                    实际使用的图片清单
```

## 在闭环或推理中使用

`summary.json` 的 `checkpoint` 和 `thresholds` 可直接交给 `detected_pipeline.plugins.yolo_supervised.YoloFeedbackAdapter`：

```python
from detected_pipeline.plugins.yolo_supervised import YoloFeedbackAdapter
adapter = YoloFeedbackAdapter(Path(summary["checkpoint"]), {**summary["thresholds"], **summary["inference"],
                              "model_version": summary["model_version"], "output_dir": "results/infer", "roi_mask": summary["roi_mask"]})
adapter.load(); prediction = adapter.predict(image_path, InferenceContext(sample_id, category))
```
