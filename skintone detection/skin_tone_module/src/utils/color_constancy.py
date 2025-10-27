"""Color constancy utilities (Gray-World)"""
import numpy as np
from PIL import Image


def gray_world(img: Image.Image) -> Image.Image:
    """Apply Gray-World color normalization to a PIL image and return normalized PIL image."""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        return img
    # compute per-channel mean
    means = arr.mean(axis=(0, 1))
    # avoid div by zero
    means = np.maximum(means, 1e-6)
    mean_gray = means.mean()
    scale = mean_gray / means
    arr = arr * scale
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
