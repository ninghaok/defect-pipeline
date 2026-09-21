from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def _read(path: str | Path) -> np.ndarray:
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(str(path))
    # cv2.imread is unreliable for Chinese/non-ASCII paths on Windows.
    # Reading bytes through NumPy and decoding them in memory is Unicode-safe.
    value=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if value is None:raise FileNotFoundError(str(path))
    return value

def external_gt(path: str | Path, shape: tuple[int,int] | None=None) -> np.ndarray:
    """Read source GT. dataset_523 is light background and dark anomaly."""
    value=_read(path)
    if shape and value.shape!=shape:value=cv2.resize(value,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return value<128

def internal_mask(path: str | Path, shape: tuple[int,int] | None=None) -> np.ndarray:
    """Read canonical/prediction mask: black background, white anomaly."""
    value=_read(path)
    if shape and value.shape!=shape:value=cv2.resize(value,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return value>=128

def write_internal_from_external(source: str | Path, target: str | Path) -> None:
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    mask=external_gt(source).astype(np.uint8)*255
    ok,data=cv2.imencode('.png',mask)
    if not ok:raise RuntimeError(f'cannot encode canonical mask: {target}')
    data.tofile(target)
