"""Decode Werk image outputs into ComfyUI IMAGE tensors.

RGBA images are composited onto an opaque white background before conversion to
RGB. Images are never silently resized.
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

try:
    from .config import DEFAULT_MAX_IMAGE_PIXELS
except ImportError:  # pragma: no cover - direct-module development
    from config import DEFAULT_MAX_IMAGE_PIXELS



def decode_base64_image(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image Base64 data must be a non-empty string")
    encoded = value.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("image data URL must contain Base64 data")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid image Base64 data") from error


def image_bytes_to_tensor(data: bytes, *, max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS) -> torch.Tensor:
    if not data:
        raise ValueError("image response is empty")
    try:
        with Image.open(BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError(f"image dimensions {width}x{height} exceed the {max_pixels}-pixel limit")
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array.copy()).unsqueeze(0)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("response is not a valid supported image") from error


def batch_image_tensors(images: Iterable[torch.Tensor]) -> torch.Tensor:
    values = list(images)
    if not values:
        raise ValueError("Werk returned no images")
    shape = tuple(values[0].shape[1:])
    if any(tuple(value.shape[1:]) != shape for value in values):
        raise ValueError("Werk returned images with different dimensions; they cannot form one ComfyUI batch")
    return torch.cat(values, dim=0).to(dtype=torch.float32)
