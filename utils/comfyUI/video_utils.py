"""Convert ComfyUI image inputs and Werk video outputs without eager Comfy imports."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch


def image_tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    """Encode exactly one ComfyUI IMAGE tensor as an opaque RGB PNG."""

    if not isinstance(image, torch.Tensor):
        raise TypeError("initial_image must be a ComfyUI IMAGE tensor")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("initial_image must contain exactly one image")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("initial_image dimensions must be greater than zero")
    channels = int(image.shape[3])
    if channels not in {1, 3, 4}:
        raise ValueError("initial_image must have 1, 3, or 4 channels")

    values = image.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    if not np.isfinite(values).all():
        raise ValueError("initial_image contains non-finite pixel values")
    values = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    if channels == 1:
        values = np.repeat(values, 3, axis=2)
    elif channels == 4:
        rgb = values[..., :3].astype(np.float32)
        alpha = values[..., 3:4].astype(np.float32) / 255.0
        values = np.rint((rgb * alpha) + (255.0 * (1.0 - alpha))).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def image_tensor_to_api_input(image: torch.Tensor) -> dict[str, str]:
    return {
        "base64": base64.b64encode(image_tensor_to_png_bytes(image)).decode("ascii"),
        "mime_type": "image/png",
    }


def video_bytes_to_comfy(
    data: bytes,
    *,
    factory: Callable[[BytesIO], Any] | None = None,
) -> Any:
    """Wrap encoded video bytes in ComfyUI's native VIDEO implementation."""

    if not isinstance(data, bytes) or not data:
        raise ValueError("video response is empty")
    if factory is None:
        try:
            from comfy_api.latest import InputImpl
        except ImportError as error:  # pragma: no cover - requires ComfyUI runtime
            raise RuntimeError(
                "native VIDEO output requires a ComfyUI version with comfy_api.latest"
            ) from error
        factory = InputImpl.VideoFromFile
    return factory(BytesIO(data))
