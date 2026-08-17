import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

from ..video_utils import (
    image_tensor_to_api_input,
    image_tensor_to_png_bytes,
    video_bytes_to_comfy,
)


def decoded_png(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        return np.asarray(image)


def test_comfy_image_tensor_encodes_as_rgb_png_api_input():
    tensor = torch.tensor(
        [[[[1.0, 0.5, 0.0], [0.0, 0.25, 1.0]]]], dtype=torch.float32
    )
    payload = image_tensor_to_api_input(tensor)
    pixels = decoded_png(base64.b64decode(payload["base64"]))
    assert payload["mime_type"] == "image/png"
    assert pixels.shape == (1, 2, 3)
    assert pixels[0, 0].tolist() == [255, 128, 0]


def test_rgba_initial_image_is_composited_on_white():
    tensor = torch.tensor([[[[1.0, 0.0, 0.0, 0.5]]]], dtype=torch.float32)
    assert decoded_png(image_tensor_to_png_bytes(tensor))[0, 0].tolist() == [255, 127, 127]


@pytest.mark.parametrize(
    "value,message",
    [
        (torch.zeros((2, 1, 1, 3)), "exactly one"),
        (torch.zeros((1, 1, 1, 2)), "1, 3, or 4"),
        (torch.tensor([[[[float("nan"), 0.0, 0.0]]]]), "non-finite"),
    ],
)
def test_invalid_initial_image_tensors_are_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        image_tensor_to_png_bytes(value)


def test_video_bytes_use_lazy_native_comfy_factory():
    seen = {}

    def factory(stream):
        seen["stream"] = stream
        return "native-video"

    assert video_bytes_to_comfy(b"encoded-video", factory=factory) == "native-video"
    assert seen["stream"].read() == b"encoded-video"
    with pytest.raises(ValueError, match="empty"):
        video_bytes_to_comfy(b"", factory=factory)
