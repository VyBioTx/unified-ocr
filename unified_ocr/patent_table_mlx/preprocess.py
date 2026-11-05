"""Image preprocessing for the MLX SLANeXt model.

Replicates the PaddleX ``table_structure_recognition`` preprocessing exactly:

    ReadImage(BGR) -> ResizeByLong(512) -> Normalize(ImageNet) -> Pad(512) -> CHW

(see ``paddlex.inference.models.table_structure_recognition.predictor``).  The
resize uses a vectorised bilinear kernel equivalent to ``cv2.INTER_LINEAR`` so no
OpenCV dependency is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_bilinear_cv2(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear resize matching ``cv2.resize(..., interpolation=INTER_LINEAR)``."""
    src_h, src_w = img.shape[:2]
    if (src_h, src_w) == (out_h, out_w):
        return img.copy()

    scale_y = src_h / out_h
    scale_x = src_w / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * scale_y - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * scale_x - 0.5

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    # clamp with border replication (OpenCV BORDER_REPLICATE)
    def clip(a, lo, hi):
        return np.clip(a, lo, hi)

    y0c = clip(y0, 0, src_h - 1)
    y1c = clip(y0 + 1, 0, src_h - 1)
    x0c = clip(x0, 0, src_w - 1)
    x1c = clip(x0 + 1, 0, src_w - 1)

    imgf = img.astype(np.float32)
    top = imgf[y0c][:, x0c] * (1 - wx) + imgf[y0c][:, x1c] * wx
    bot = imgf[y1c][:, x0c] * (1 - wx) + imgf[y1c][:, x1c] * wx
    out = top * (1 - wy) + bot * wy
    return out


def preprocess_image(
    image: str | Path | np.ndarray,
    image_size: int = 512,
    mean: tuple = _IMAGENET_MEAN,
    std: tuple = _IMAGENET_STD,
    scale_255: bool = True,
) -> np.ndarray:
    """Return an ``float32`` CHW tensor ``[3, 512, 512]`` ready for :class:`SLANeXt`.

    Args:
        image: path to an image, or an HWC uint8 array assumed to be **BGR**
            (matching PaddleX ``ReadImage(format="BGR")``).
    """
    if isinstance(image, (str, Path)):
        from PIL import Image

        arr = np.array(Image.open(image).convert("RGB"))
        arr = arr[..., ::-1]  # RGB -> BGR
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HWC 3-channel image, got shape {arr.shape}")

    h, w = arr.shape[:2]
    scale = image_size / max(h, w)
    out_h, out_w = round(h * scale), round(w * scale)
    resized = _resize_bilinear_cv2(arr, out_w, out_h)

    resized = resized.astype(np.float32)
    if scale_255:
        resized = resized / 255.0
    alpha = np.array([1.0 / s for s in std], dtype=np.float32)
    beta = np.array([-m / s for m, s in zip(mean, std)], dtype=np.float32)
    resized = resized * alpha + beta

    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:out_h, :out_w] = resized
    return canvas.transpose(2, 0, 1)  # CHW
