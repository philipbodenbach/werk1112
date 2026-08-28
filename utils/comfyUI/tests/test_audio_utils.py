import base64
from io import BytesIO
import struct
import wave

import numpy as np
import pytest
import torch

from ..audio_utils import audio_bytes_to_comfy, comfy_audio_to_api_input


def wav_bytes(*, channels=2, sample_rate=8_000, frames=17):
    samples = [((index * 401) % 20_000) - 10_000 for index in range(frames * channels)]
    output = BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


def flac_bytes(*, channels=2, sample_rate=8_000, frames=1152):
    av = pytest.importorskip("av")
    layout = "mono" if channels == 1 else "stereo"
    samples = np.arange(frames * channels, dtype=np.int16).reshape(1, -1)
    output = BytesIO()
    container = av.open(output, mode="w", format="flac")
    stream = container.add_stream("flac", rate=sample_rate)
    stream.layout = layout
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout=layout)
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


@pytest.mark.parametrize(
    "encoded,channels,frames",
    [
        (lambda: wav_bytes(channels=1), 1, 17),
        (lambda: wav_bytes(channels=2), 2, 17),
        (lambda: flac_bytes(channels=2), 2, 1152),
    ],
)
def test_audio_bytes_to_native_comfy_audio(encoded, channels, frames):
    value = audio_bytes_to_comfy(encoded())
    assert value["sample_rate"] == 8_000
    assert value["waveform"].shape == (1, channels, frames)
    assert value["waveform"].dtype == torch.float32
    assert torch.isfinite(value["waveform"]).all()


def test_audio_decoder_accepts_planar_frame_shapes():
    class Codec:
        sample_rate = 16_000
        channels = 2

    class Stream:
        codec_context = Codec()

    class Frame:
        def to_ndarray(self):
            return np.array([[0.25, -0.5, 1.0], [-1.0, 0.5, 0.0]], dtype=np.float32)

    class Streams:
        audio = [Stream()]

    class Container:
        streams = Streams()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, stream):
            assert isinstance(stream, Stream)
            return iter([Frame()])

    value = audio_bytes_to_comfy(
        b"encoded", open_container=lambda *_args, **_kwargs: Container()
    )
    assert value["waveform"].shape == (1, 2, 3)
    torch.testing.assert_close(
        value["waveform"][0],
        torch.tensor([[0.25, -0.5, 1.0], [-1.0, 0.5, 0.0]]),
    )


def test_native_audio_input_is_pcm16_wav_base64_and_round_trips():
    source = {
        "waveform": torch.tensor(
            [[[0.0, 0.5, -0.5], [1.0, -1.0, 0.25]]], dtype=torch.float32
        ),
        "sample_rate": 24_000,
    }
    api_input = comfy_audio_to_api_input(source)
    assert api_input["modality"] == "audio"
    assert api_input["role"] == "input_audio"
    assert api_input["source"]["kind"] == "base64"
    assert api_input["mime_type"] == "audio/wav"

    encoded = base64.b64decode(api_input["source"]["data"])
    decoded = audio_bytes_to_comfy(encoded)
    assert decoded["sample_rate"] == 24_000
    assert decoded["waveform"].shape == (1, 2, 3)
    torch.testing.assert_close(
        decoded["waveform"], source["waveform"], atol=4e-5, rtol=0
    )


@pytest.mark.parametrize(
    "value,message",
    [
        (b"", "empty"),
        (b"not audio", "valid supported audio"),
    ],
)
def test_audio_decoder_rejects_invalid_payloads(value, message):
    with pytest.raises(ValueError, match=message):
        audio_bytes_to_comfy(value)


def test_audio_encoder_rejects_batches_and_nonfinite_samples():
    with pytest.raises(ValueError, match="exactly one batch"):
        comfy_audio_to_api_input(
            {"waveform": torch.zeros((2, 1, 2)), "sample_rate": 8_000}
        )
    with pytest.raises(ValueError, match="non-finite"):
        comfy_audio_to_api_input(
            {"waveform": torch.tensor([[[float("nan")]]]), "sample_rate": 8_000}
        )


def test_audio_encoder_rejects_input_just_over_byte_limit_before_allocating():
    audio = {"waveform": torch.zeros((1, 2, 10)), "sample_rate": 8_000}
    # Canonical PCM WAV estimate: 44-byte header + 2 channels * 10 * int16.
    with pytest.raises(ValueError, match="exceeds 83 bytes"):
        comfy_audio_to_api_input(audio, max_bytes=83)
