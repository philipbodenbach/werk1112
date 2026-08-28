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


def comfy_images_to_data_urls(
    images: torch.Tensor,
    *,
    max_pixels: int,
    max_bytes: int,
) -> list[str]:
    """Encode an ordered ComfyUI IMAGE batch as bounded RGB PNG data URLs."""

    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a ComfyUI IMAGE tensor")
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("images must contain at least one ComfyUI image")
    batch, height, width, channels = (int(value) for value in images.shape)
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be greater than zero")
    if channels not in {1, 3, 4}:
        raise ValueError("images must have 1, 3, or 4 channels")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be greater than zero")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    total_pixels = batch * height * width
    if total_pixels > max_pixels:
        raise ValueError(
            f"vision image batch contains {total_pixels} pixels and exceeds "
            f"the {max_pixels}-pixel limit"
        )

    values = images.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.isfinite(values).all():
        raise ValueError("images contain non-finite pixel values")
    values = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)

    encoded_images: list[bytes] = []
    total_bytes = 0
    for values_for_image in values:
        if channels == 1:
            rgb = np.repeat(values_for_image, 3, axis=2)
        elif channels == 4:
            color = values_for_image[..., :3].astype(np.float32)
            alpha = values_for_image[..., 3:4].astype(np.float32) / 255.0
            rgb = np.rint(
                (color * alpha) + (255.0 * (1.0 - alpha))
            ).astype(np.uint8)
        else:
            rgb = values_for_image
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        encoded = buffer.getvalue()
        total_bytes += len(encoded)
        if total_bytes > max_bytes:
            raise ValueError(
                f"encoded vision image batch exceeds {max_bytes} bytes"
            )
        encoded_images.append(encoded)

    return [
        "data:image/png;base64," + base64.b64encode(value).decode("ascii")
        for value in encoded_images
    ]
