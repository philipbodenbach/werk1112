import base64
from io import BytesIO

from PIL import Image
import pytest
import torch

from ..image_utils import (
    batch_image_tensors,
    comfy_images_to_data_urls,
    decode_base64_image,
    image_bytes_to_tensor,
)


def png_bytes(mode="RGB", size=(2, 3), color=None):
    if color is None:
        color = (20, 40, 60) if mode == "RGB" else 80
    image = Image.new(mode, size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_raw_and_data_url_base64_decode():
    raw = png_bytes()
    encoded = base64.b64encode(raw).decode()
    assert decode_base64_image(encoded) == raw
    assert decode_base64_image(f"data:image/png;base64,{encoded}") == raw


def test_invalid_base64_fails():
    with pytest.raises(ValueError, match="invalid image Base64"):
        decode_base64_image("not base64!!!")


def test_rgb_tensor_shape_dtype_and_range():
    tensor = image_bytes_to_tensor(png_bytes(size=(2, 3)))
    assert tensor.shape == (1, 3, 2, 3)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_rgba_is_composited_on_white():
    tensor = image_bytes_to_tensor(png_bytes("RGBA", (1, 1), (255, 0, 0, 0)))
    assert torch.allclose(tensor[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]))


def test_grayscale_is_converted_to_rgb():
    tensor = image_bytes_to_tensor(png_bytes("L", (1, 1), 128))
    assert tensor.shape == (1, 1, 1, 3)
    assert torch.allclose(tensor[0, 0, 0, 0], tensor[0, 0, 0, 1])


def test_oversized_image_is_rejected_before_conversion():
    with pytest.raises(ValueError, match="pixel limit"):
        image_bytes_to_tensor(png_bytes(size=(20, 20)), max_pixels=399)


def test_multiple_images_form_batch():
    images = [image_bytes_to_tensor(png_bytes(size=(2, 3))) for _ in range(2)]
    assert batch_image_tensors(images).shape == (2, 3, 2, 3)


def test_mismatched_dimensions_fail_without_resize():
    images = [
        image_bytes_to_tensor(png_bytes(size=(2, 3))),
        image_bytes_to_tensor(png_bytes(size=(3, 3))),
    ]
    with pytest.raises(ValueError, match="different dimensions"):
        batch_image_tensors(images)


def test_comfy_image_batch_encodes_ordered_bounded_png_data_urls():
    images = torch.stack(
        [
            torch.full((2, 3, 3), 0.1),
            torch.full((2, 3, 3), 0.9),
        ]
    )
    values = comfy_images_to_data_urls(images, max_pixels=12, max_bytes=4096)
    assert len(values) == 2
    decoded = [Image.open(BytesIO(decode_base64_image(value))).getpixel((0, 0)) for value in values]
    assert decoded == [(26, 26, 26), (230, 230, 230)]


def test_comfy_image_batch_limits_aggregate_pixels_and_encoded_bytes():
    images = torch.zeros((2, 2, 3, 3))
    with pytest.raises(ValueError, match="pixel limit"):
        comfy_images_to_data_urls(images, max_pixels=11, max_bytes=4096)
    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        comfy_images_to_data_urls(images, max_pixels=12, max_bytes=1)


def test_comfy_image_batch_rejects_invalid_shape_channels_and_values():
    with pytest.raises(ValueError, match="at least one"):
        comfy_images_to_data_urls(torch.zeros((0, 2, 3, 3)), max_pixels=10, max_bytes=10)
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        comfy_images_to_data_urls(torch.zeros((1, 2, 3, 2)), max_pixels=10, max_bytes=10)
    invalid = torch.zeros((1, 1, 1, 3))
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        comfy_images_to_data_urls(invalid, max_pixels=10, max_bytes=100)
