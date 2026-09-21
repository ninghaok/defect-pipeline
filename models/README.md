# 模型权重（不入库）

运行前把以下三个文件放到本目录：

| 文件 | 用途 | 来源 |
|---|---|---|
| `yolo26s-seg.pt` | YOLO 实例分割基座 | Ultralytics yolo26s-seg 官方权重 |
| `dinov2_vitl14_pretrain.pth` | DINOv2 ViT-L/14 骨干 | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth |
| `checkpoints_pro_angle.pth` | ADPretrain 角度投影器（dinov2-large 版） | ADPretrain 官方发布（NeurIPS 2025） |

`python cli/check_local_installation.py` 会检查这三个文件是否存在。
