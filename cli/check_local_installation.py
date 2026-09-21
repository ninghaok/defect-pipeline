from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    import cv2, numpy, torch, torchvision, yaml, ultralytics
    required = [PROJECT / "models/dinov2_vitl14_pretrain.pth", PROJECT / "models/checkpoints_pro_angle.pth",
                PROJECT / "models/yolo26s-seg.pt", PROJECT / "configs/pretrained.yaml", PROJECT / "roi/qiusaiwaiyuan.png",
                PROJECT / "vendor/adpretrain/models/dino.py"]
    missing = [str(path) for path in required if not path.is_file()]
    payload = {"python": sys.executable, "torch": torch.__version__, "torchvision": torchvision.__version__,
               "ultralytics": ultralytics.__version__, "cuda": torch.cuda.is_available(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "opencv": cv2.__version__, "numpy": numpy.__version__, "yaml": yaml.__version__, "missing": missing}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit("required local resources are missing")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")


if __name__ == "__main__":
    main()
