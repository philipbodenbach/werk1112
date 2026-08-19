import base64
from io import BytesIO
import json
from pathlib import Path
import struct
import wave

import pytest
import torch

from ..audio_utils import DEFAULT_MAX_AUDIO_INPUT_BYTES as UTILS_AUDIO_INPUT_LIMIT
from ..config import (
    DEFAULT_MAX_AUDIO_INPUT_BYTES,
    WerkAudioConfig,
    WerkConnection,
    environment_max_audio_input_bytes,
)
from .. import nodes
from ..nodes import (
    AUDIO_ANALYSIS_TASKS,
    AUDIO_DETECTION_TASKS,
    AUDIO_EMBEDDING_TASKS,
    AUDIO_GENERATION_TASKS,
    AUDIO_TASKS,
    AUDIO_TEXT_OUTPUT_TASKS,
    AUDIO_TRANSCRIPTION_TASKS,
    AUDIO_TRANSFORM_TASKS,
    WerkAudioAnalyzeNode,
    WerkAudioConfigNode,
    WerkAudioGenerateNode,
    WerkAudioModelsNode,
    WerkAudioParametersNode,
    WerkAudioProcessNode,
    audio_config_payload,
    build_audio_config,
    build_audio_input_job_request,
    build_configured_audio_request,
    classify_audio_models,
    normalize_audio_job_parameters,
)


def native_audio():
    return {
        "waveform": torch.tensor([[[0.0, 0.25, -0.25, 0.5]]]),
        "sample_rate": 16_000,
    }


def wav_bytes():
    output = BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16_000)
        destination.writeframes(struct.pack("<4h", 0, 1000, -1000, 2000))
    return output.getvalue()


def models_payload(*ids):
    return {"data": [{"id": value} for value in ids]}


def completed_job(*, output_id="out-audio", mime_type="audio/wav"):
    return {
        "id": "job-audio",
        "status": "completed",
        "created_unix": 1,
        "updated_unix": 2,
        "request": {"inputs": [{"source": {"kind": "base64", "data": "secret"}}]},
        "result": {
            "id": "result-audio",
            "task": "audio_generation",
            "model": "audio-model",
            "runtime": "test",
            "effective_request": {
                "inputs": [{"source": {"kind": "base64", "data": "effective-secret"}}]
            },
            "outputs": [{"id": output_id, "mime_type": mime_type}],
        },
    }


class FakeClient:
    responses = {}
    downloads = {}
    posted = []
    limits = []

    def __init__(self, _connection):
        pass

    def get_json(self, path, query=None):
        value = self.responses[path]
        return value(query) if callable(value) else value

    def post_json(self, path, payload):
        self.__class__.posted.append((path, payload))
        value = self.responses[path]
        return value(payload) if callable(value) else value

    def delete_json(self, path):
        return {"id": path.rsplit("/", 1)[-1], "status": "cancelled"}

    def download_bytes(self, path, *, max_bytes=None):
        self.__class__.limits.append(max_bytes)
        return self.downloads[path]


@pytest.fixture
def fake_client(monkeypatch):
    FakeClient.responses = {}
    FakeClient.downloads = {}
    FakeClient.posted = []
    FakeClient.limits = []
    monkeypatch.setattr(nodes, "WerkClient", FakeClient)
    return FakeClient


def test_audio_taxonomy_matches_the_rust_cli_groups_exactly():
    assert AUDIO_GENERATION_TASKS == (
        "audio-generation",
        "music-generation",
        "text-to-speech",
    )
    assert AUDIO_TRANSCRIPTION_TASKS == ("speech-to-text", "speech-translation")
    assert AUDIO_DETECTION_TASKS == (
        "audio-event-detection",
        "voice-activity-detection",
        "speaker-identification",
        "language-identification",
        "speech-emotion-recognition",
    )
    assert AUDIO_ANALYSIS_TASKS == (
        "audio-captioning",
        "speaker-diarization",
        "audio-classification",
        "audio-understanding",
    )
    assert AUDIO_TRANSFORM_TASKS == (
        "voice-conversion",
        "stem-separation",
        "audio-enhancement",
        "audio-editing",
    )
    assert AUDIO_EMBEDDING_TASKS == ("audio-embedding",)
    assert len(AUDIO_TASKS) == 19
    assert "song-continuation" not in AUDIO_TASKS


def test_audio_input_limit_default_and_environment_override_are_consistent(monkeypatch):
    assert DEFAULT_MAX_AUDIO_INPUT_BYTES == 64 * 1024 * 1024
    assert UTILS_AUDIO_INPUT_LIMIT == DEFAULT_MAX_AUDIO_INPUT_BYTES
    monkeypatch.setenv("WERK_MAX_AUDIO_INPUT_BYTES", "1234")
    assert environment_max_audio_input_bytes() == 1234
    monkeypatch.setenv("WERK_MAX_AUDIO_INPUT_BYTES", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        environment_max_audio_input_bytes()


def test_audio_model_classification_and_selection_are_task_specific(fake_client):
    capabilities = {
        "models": [
            {
                "id": "transcriber",
                "tasks": ["speech_to_text", "speech_translation"],
                "available_tasks": ["speech_to_text"],
            },
            {
                "id": "separator",
                "tasks": ["stem_separation"],
                "available_tasks": ["stem_separation"],
            },
        ]
    }
    result = classify_audio_models(
        models_payload("transcriber", "separator", "image"), capabilities
    )
    assert result["by_task"]["speech-to-text"]["available"] == ["transcriber"]
    assert result["by_task"]["speech-translation"]["declared"] == ["transcriber"]
    assert result["by_task"]["speech-translation"]["available"] == []
    assert result["by_task"]["stem-separation"]["available"] == ["separator"]

    fake_client.responses = {
        "/v1/models": models_payload("transcriber", "separator"),
        "/v1/capabilities": capabilities,
    }
    selected, choices, metadata = WerkAudioModelsNode().select(
        WerkConnection("http://werk"), "stem-separation", 0, "", True
    )
    assert selected == "separator"
    assert choices == "separator"
    assert json.loads(metadata)["selected_task"] == "stem-separation"


def test_audio_parameters_forwards_every_selected_task(fake_client):
    seen = {}

    def response(query):
        seen.update(query)
        return {"parameters": [{"path": "audio.top_k"}]}

    fake_client.responses["/v1/parameters"] = response
    payload, summary = WerkAudioParametersNode().parameters(
        WerkConnection("http://werk"),
        "classifier",
        "audio-event-detection",
        "auto",
        0,
    )
    assert seen == {
        "task": "audio-event-detection",
        "model": "classifier",
        "backend": "auto",
    }
    assert json.loads(payload)["parameters"][0]["path"] == "audio.top_k"
    assert "audio-event-detection" in summary


def test_audio_configs_are_typed_and_tts_default_does_not_send_seed():
    config, payload = WerkAudioConfigNode().configure(
        task="music-generation",
        duration=12.5,
        variations=2,
        seed=42,
        sample_rate=48_000,
        channels=2,
        output_format="flac",
        instrumental="enabled",
        voice="",
        speed=1.0,
        additional_audio_parameters_json='{"guidance":3.5}',
    )
    assert isinstance(config, WerkAudioConfig)
    assert json.loads(payload) == audio_config_payload(config)
    assert dict(config.request_fields) == {"response_format": "flac", "n": 2}
    assert dict(config.parameters) == {
        "audio.duration": 12.5,
        "audio.seed": 42,
        "audio.instrumental": True,
        "audio.sample_rate": 48_000,
        "audio.channels": 2,
        "audio.guidance": 3.5,
    }

    tts = build_audio_config(task="text-to-speech")
    assert dict(tts.parameters) == {}
    request = build_configured_audio_request(
        task="text-to-speech", model="tts", prompt="Hallo", config=tts
    )
    assert request["input"] == "Hallo"
    assert request["async"] is True
    assert request["parameters"] == {}
    assert "task" not in request


def test_generic_audio_input_job_uses_bounded_embedded_wav_and_task_namespace(
    monkeypatch,
):
    monkeypatch.setenv("WERK_MAX_AUDIO_INPUT_BYTES", "1024")
    request = build_audio_input_job_request(
        task="speech-translation",
        model="whisper",
        audio=native_audio(),
        prompt="names: Werk",
        additional_audio_parameters_json='{"language":"de"}',
    )
    assert request["task"] == "speech-translation"
    assert request["parameters"] == {"stt.language": "de"}
    source = request["inputs"][0]
    assert source["modality"] == "audio"
    assert source["source"]["kind"] == "base64"
    assert base64.b64decode(source["source"]["data"]).startswith(b"RIFF")
    assert normalize_audio_job_parameters('{"top_k":5}', "audio-classification") == {
        "audio.top_k": 5
    }

    monkeypatch.setenv("WERK_MAX_AUDIO_INPUT_BYTES", "51")
    with pytest.raises(ValueError, match="exceeds 51 bytes"):
        build_audio_input_job_request(
            task="speech-to-text", model="whisper", audio=native_audio()
        )


def test_voice_conversion_accepts_reference_audio_and_required_prompts_are_clear():
    request = build_audio_input_job_request(
        task="voice-conversion",
        model="rvc",
        audio=native_audio(),
        reference_audio=native_audio(),
    )
    assert [item["role"] for item in request["inputs"]] == [
        "input_audio",
        "reference_audio",
    ]
    with pytest.raises(ValueError, match="only for voice-conversion"):
        build_audio_input_job_request(
            task="audio-enhancement",
            model="enhancer",
            audio=native_audio(),
            reference_audio=native_audio(),
        )
    for task in ("audio-understanding", "audio-editing"):
        with pytest.raises(ValueError, match=f"empty for {task}"):
            build_audio_input_job_request(
                task=task,
                model="model",
                audio=native_audio(),
            )


def test_audio_generate_posts_task_specific_endpoint_and_returns_native_audio(
    fake_client,
):
    fake_client.responses["/v1/audio/generations"] = completed_job()
    fake_client.downloads["/v1/outputs/out-audio"] = (wav_bytes(), "audio/wav")
    result = WerkAudioGenerateNode().generate(
        WerkConnection("http://werk"),
        "music-model",
        "music-generation",
        "soft piano",
        "noise",
        build_audio_config(task="music-generation", seed=7),
    )
    assert result[0][0]["waveform"].shape == (1, 1, 4)
    assert result[2:] == (7, "job-audio", "result-audio", "out-audio")
    endpoint, request = fake_client.posted[0]
    assert endpoint == "/v1/audio/generations"
    assert request["task"] == "music-generation"
    assert request["negative_prompt"] == "noise"
    assert "secret" not in result[1]


def test_audio_process_and_analysis_use_generic_jobs_and_bounded_outputs(fake_client):
    fake_client.responses["/v1/jobs"] = completed_job()
    fake_client.downloads["/v1/outputs/out-audio"] = (wav_bytes(), "audio/wav")
    processed = WerkAudioProcessNode().process(
        WerkConnection("http://werk"),
        "enhancer",
        "audio-enhancement",
        native_audio(),
        "",
        "",
        "{}",
    )
    assert processed[0][0]["waveform"].shape == (1, 1, 4)
    assert fake_client.posted[0][0] == "/v1/jobs"
    assert fake_client.posted[0][1]["task"] == "audio-enhancement"

    fake_client.responses["/v1/jobs"] = completed_job(
        output_id="out-json", mime_type="application/json"
    )
    fake_client.downloads["/v1/outputs/out-json"] = (
        b'{"text":"a dog bark","score":0.9}',
        "application/json; charset=utf-8",
    )
    analyzed = WerkAudioAnalyzeNode().analyze(
        WerkConnection("http://werk"),
        "captioner",
        "audio-captioning",
        native_audio(),
        "",
        "{}",
    )
    assert json.loads(analyzed[0][0]) == {"text": "a dog bark", "score": 0.9}
    assert fake_client.posted[-1][1]["task"] == "audio-captioning"
    assert fake_client.limits[-1] == nodes.MAX_AUDIO_TEXT_BYTES


def test_audio_node_native_types_and_task_scopes_are_explicit():
    assert WerkAudioGenerateNode.RETURN_TYPES[0] == "AUDIO"
    assert WerkAudioProcessNode.RETURN_TYPES[0] == "AUDIO"
    assert WerkAudioAnalyzeNode.RETURN_TYPES[0] == "STRING"
    assert WerkAudioGenerateNode.OUTPUT_IS_LIST[0] is True
    assert WerkAudioProcessNode.OUTPUT_IS_LIST[0] is True
    assert WerkAudioAnalyzeNode.OUTPUT_IS_LIST[0] is True
    assert WerkAudioGenerateNode.INPUT_TYPES()["required"]["task"][0] == list(
        AUDIO_GENERATION_TASKS
    )
    assert WerkAudioProcessNode.INPUT_TYPES()["required"]["task"][0] == list(
        AUDIO_TRANSFORM_TASKS
    )
    assert (
        WerkAudioProcessNode.INPUT_TYPES()["optional"]["reference_audio"][0] == "AUDIO"
    )
    assert WerkAudioAnalyzeNode.INPUT_TYPES()["required"]["task"][0] == list(
        AUDIO_TEXT_OUTPUT_TASKS
    )


def test_audio_api_prompt_examples_are_valid_and_keep_task_links_explicit():
    examples = Path(__file__).parents[1] / "examples"
    music = json.loads((examples / "werk_music_generation_api.json").read_text())
    tts = json.loads((examples / "werk_text_to_speech_api.json").read_text())
    understanding = json.loads(
        (examples / "werk_audio_understanding_api.json").read_text()
    )
    conversion = json.loads((examples / "werk_voice_conversion_api.json").read_text())

    assert music["2"]["inputs"]["task"] == "music-generation"
    assert music["4"]["class_type"] == "WerkAudioConfig"
    assert music["5"]["inputs"]["model"] == ["2", 0]
    assert music["6"]["class_type"] == "PreviewAudio"
    assert tts["3"]["inputs"]["task"] == "text-to-speech"
    assert tts["4"]["inputs"]["config"] == ["3", 0]
    assert understanding["4"]["class_type"] == "WerkAudioAnalyze"
    assert understanding["4"]["inputs"]["source_audio"] == ["3", 0]
    assert understanding["4"]["inputs"]["prompt"]
    assert conversion["5"]["class_type"] == "WerkAudioProcess"
    assert conversion["5"]["inputs"]["source_audio"] == ["3", 0]
    assert conversion["5"]["inputs"]["reference_audio"] == ["4", 0]
    assert conversion["6"]["inputs"]["audio"] == ["5", 0]
