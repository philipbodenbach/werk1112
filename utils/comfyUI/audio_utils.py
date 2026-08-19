"""Decode Werk audio artifacts into ComfyUI's native ``AUDIO`` value."""

from __future__ import annotations

import base64
from io import BytesIO
import wave
from typing import Any, Callable

import torch

DEFAULT_MAX_AUDIO_INPUT_BYTES = 64 * 1024 * 1024


def _pcm_to_float32(value: torch.Tensor) -> torch.Tensor:
    if value.dtype.is_floating_point:
        return value.to(dtype=torch.float32)
    if value.dtype == torch.bool:
        raise ValueError("decoded audio uses an unsupported boolean sample format")
    try:
        limits = torch.iinfo(value.dtype)
    except TypeError as error:
        raise ValueError(
            f"decoded audio uses unsupported dtype {value.dtype}"
        ) from error
    samples = value.to(dtype=torch.float32)
    if limits.min == 0:
        midpoint = float(limits.max + 1) / 2.0
        return (samples - midpoint) / midpoint
    scale = float(max(abs(limits.min), limits.max))
    return samples / scale


def audio_bytes_to_comfy(
    data: bytes,
    *,
    open_container: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return ComfyUI's native ``AUDIO`` dictionary from encoded audio bytes.

    ComfyUI 0.29.0 represents audio as a float32 waveform shaped ``[B,C,T]``
    plus an integer sample rate.  Its built-in Load Audio node also decodes via
    PyAV, so using the same decoder keeps WAV, FLAC, and OGG behavior aligned.
    """

    if not isinstance(data, bytes) or not data:
        raise ValueError("audio response is empty")
    if open_container is None:
        try:
            import av
        except ImportError as error:  # pragma: no cover - ComfyUI depends on PyAV
            raise RuntimeError(
                "native AUDIO output requires ComfyUI's PyAV dependency"
            ) from error
        open_container = av.open

    try:
        with open_container(BytesIO(data), mode="r") as container:
            streams = list(container.streams.audio)
            if not streams:
                raise ValueError("audio response contains no audio stream")
            stream = streams[0]
            codec_context = getattr(stream, "codec_context", None)
            sample_rate = int(
                getattr(codec_context, "sample_rate", 0)
                or getattr(stream, "sample_rate", 0)
                or 0
            )
            channels = int(
                getattr(codec_context, "channels", 0)
                or getattr(stream, "channels", 0)
                or 0
            )
            if sample_rate <= 0 or channels <= 0:
                raise ValueError("audio response has invalid stream metadata")

            frames: list[torch.Tensor] = []
            # Passing the stream object is supported across the PyAV versions
            # shipped by ComfyUI and avoids the version-sensitive ``streams=``
            # keyword form.
            for frame in container.decode(stream):
                samples = torch.from_numpy(frame.to_ndarray())
                if samples.ndim == 1:
                    samples = samples.unsqueeze(0)
                if samples.ndim != 2:
                    raise ValueError(
                        "decoded audio frame must be one- or two-dimensional"
                    )
                if samples.shape[0] != channels:
                    if samples.numel() % channels:
                        raise ValueError(
                            "decoded audio frame has an invalid channel layout"
                        )
                    samples = samples.reshape(-1, channels).transpose(0, 1)
                if samples.shape[0] != channels:
                    raise ValueError("decoded audio frame channel count changed")
                frames.append(_pcm_to_float32(samples))

            if not frames:
                raise ValueError("audio response contains no decodable frames")
            waveform = torch.cat(frames, dim=1).unsqueeze(0).contiguous()
            if not bool(torch.isfinite(waveform).all()):
                raise ValueError("decoded audio contains non-finite samples")
            return {"waveform": waveform, "sample_rate": sample_rate}
    except ValueError as error:
        # PyAV decoding errors inherit from ValueError.  Preserve only our own
        # validation messages; normalize codec/container failures so no local
        # decoder details leak into a ComfyUI workflow error.
        if type(error) is ValueError:
            raise
        raise ValueError("response is not valid supported audio") from error
    except Exception as error:
        raise ValueError("response is not valid supported audio") from error


def comfy_audio_to_api_input(
    audio: Any,
    *,
    max_bytes: int = DEFAULT_MAX_AUDIO_INPUT_BYTES,
    role: str = "input_audio",
) -> dict[str, Any]:
    """Encode one native ComfyUI ``AUDIO`` value as an embedded PCM WAV input."""

    if not isinstance(audio, dict):
        raise TypeError("audio must be a ComfyUI AUDIO dictionary")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("audio waveform must be a torch.Tensor")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("audio sample_rate must be an integer")
    if sample_rate <= 0 or sample_rate > 384_000:
        raise ValueError("audio sample_rate must be between 1 and 384000")
    if waveform.ndim != 3:
        raise ValueError("audio waveform must have shape [batch, channels, samples]")
    if waveform.shape[0] != 1:
        raise ValueError("audio input must contain exactly one batch item")
    channels = int(waveform.shape[1])
    samples = int(waveform.shape[2])
    if not 1 <= channels <= 32:
        raise ValueError("audio waveform must contain between 1 and 32 channels")
    if samples <= 0:
        raise ValueError("audio waveform must contain at least one sample")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("audio input byte limit must be a positive integer")
    if role not in {"input_audio", "reference_audio"}:
        raise ValueError("audio input role must be input_audio or reference_audio")
    estimated_bytes = 44 + (channels * samples * 2)
    if estimated_bytes > max_bytes:
        raise ValueError(f"encoded audio input exceeds {max_bytes} bytes")

    value = waveform.detach().to(device="cpu", dtype=torch.float32)[0]
    if not bool(torch.isfinite(value).all()):
        raise ValueError("audio waveform contains non-finite samples")
    # WAV stores frames interleaved by channel.  PCM16 is universally accepted
    # by Werk media inputs and bounds the JSON expansion for long source audio.
    pcm = (
        value.clamp(-1.0, 1.0)
        .transpose(0, 1)
        .contiguous()
        .mul(32767.0)
        .round()
        .to(dtype=torch.int16)
        .numpy()
        .tobytes()
    )
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    if buffer.tell() > max_bytes:
        raise ValueError(f"encoded audio input exceeds {max_bytes} bytes")
    return {
        "modality": "audio",
        "role": role,
        "source": {
            "kind": "base64",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        },
        "mime_type": "audio/wav",
    }
