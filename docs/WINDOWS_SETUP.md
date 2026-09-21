# Windows 环境与运行

## 目录

```text
E:\ninghao\pipeline                 工程
D:\dataset_523\dataset_523          数据集：<class>\{train,val,test}\{OK,NG}，<class>\mask\<stem>_t.bmp（黑色为缺陷）
E:\ninghao\pipeline\models          yolo26s-seg.pt、dinov2_vitl14_pretrain.pth、checkpoints_pro_angle.pth
```

## 环境

```powershell
cd E:\ninghao\pipeline
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows_pipeline.ps1       # 创建 conda 环境 pipeline（torch cu128、ultralytics 8.4.115 等）
conda activate pipeline
python .\cli\check_local_installation.py   # cuda: true, missing: []
```

不再需要 faiss；最近邻用 torch 在 GPU 上精确计算。

## 运行闭环

```powershell
conda activate pipeline
cd E:\ninghao\pipeline
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
.\scripts\run_lifecycle_windows.ps1 -RunName lifecycle_seg_01
```

数据集的 `test` 目录是**固定测试集**：不流入、不训练、不标定、不参与任何选择，只用来评测每一个进入产线的模型
（预训练每次重标定后、每个 YOLO 候选训练后都在其上评测，结果写入 `workspace\test_reports\<category>\`，`test_history.jsonl` 汇总）。
流入数据 = `train` + `val` 去重后的全部可用 OK 和 NG（保留的参考、记忆库、校准 OK 除外），OK 与 NG 按比例均匀分到每个批次，
每批推理结束后自动按数据集路径（OK/NG 目录与 mask）完成审核与标注，模拟人工复核。

NG 比例是可选参数（NG 占流入总量的比例，-1 = 全部 NG）：

```powershell
.\scripts\run_lifecycle_windows.ps1 -RunName lifecycle_r10 -QiumianFupaiNgRatio 0.10 -QiumianXiepaiNgRatio 0.20 -DiMianNgRatio 0.10 -WaiYuanNgRatio 0.05
```

不同比例对应的 NG 流入张数（NG = round(OK × r / (1 − r))，超过可用数则封顶）：

| 类别 | 流内 OK | 可用 NG | 比例 0.02 | 比例 0.05 | 比例 0.10 | 比例 0.15 | 比例 0.20 | 比例 0.30 | 全部（默认） | 固定测试 OK/NG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 俯拍 | 734 | 131 | 15 | 39 | 82 | 130 | 131（封顶） | 131（封顶） | 131（15.1%） | 499/56 |
| 斜拍 | 173 | 140 | 4 | 9 | 19 | 31 | 43 | 74 | 140（44.7%） | 173/60 |
| 底面 | 311 | 144 | 6 | 16 | 35 | 55 | 78 | 133 | 144（31.6%） | 319/61 |
| 外圆 | 962 | 504 | 20 | 51 | 107 | 170 | 240 | 412 | 504（34.4%） | 597/215 |

YOLO 首次训练需要 40 张有效 NG，之后每 20 张（外圆每 40 张）一次；比例低于 0.05 时大多数类别到不了 40 张，只能验证预训练阶段。
重复同一命令会跳过已处理图片续跑；改比例必须换 `RunName`。只跑一类：加 `-Categories di_mian_detection`。

每类耗时：预训练建库约 1 分钟，推理每张约 45 ms；每个 YOLO 里程碑训练约 40 到 60 分钟（RTX 5080）。

## 单独跑消融实验

```powershell
.\scripts\run_seg_ng_ablation_live.ps1
```

## 单元测试

```powershell
python -m pytest -q tests
```

## 限制

1. 模型代码不读取 `NG` 路径、标签或 mask 来标定阈值；文件夹真值只在审核阶段使用。
2. 结果目录包含 SQLite 状态，不要只删部分文件后续跑；重做实验请换新的 `RunName`。
3. 显存不足时把 `configs/training.yaml` 的 `batch` 改为 2，其余参数不动。
