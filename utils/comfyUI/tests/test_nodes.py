import base64
import importlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest
import torch

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
from ..config import (
    WerkConnection,
    WerkImageConfig,
    WerkRoutingConfig,
    WerkVideoConfig,
)
from .. import nodes
from ..nodes import (
    WerkConnectionNode,
    WerkImageConfigNode,
    WerkImageGenerateNode,
    WerkImageModelsNode,
    WerkImageParametersNode,
    WerkRoutingConfigNode,
    WerkServerInfoNode,
    WerkVideoConfigNode,
    WerkVideoGenerateNode,
    WerkVideoModelsNode,
    WerkVideoParametersNode,
    build_configured_image_request,
    build_configured_video_request,
    build_image_config,
    build_routing_config,
    build_video_config,
    classify_image_models,
    classify_video_models,
    image_config_payload,
    normalize_image_config_parameters,
    normalize_routing_config_parameters,
    normalize_video_config_parameters,
    routing_config_payload,
    video_config_payload,
    wait_for_video_job,
)


def image_bytes(size=(2, 2)):
    buffer = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def models_payload(*ids):
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}


def capabilities_payload():
    return {
        "object": "werk.capabilities",
        "models": [
            {"id": "image-ready", "tasks": ["image_generation"], "available_tasks": ["image_generation"]},
            {"id": "image-unavailable", "tasks": ["image_generation"], "available_tasks": []},
            {"id": "text", "tasks": ["text_generation"], "available_tasks": ["text_generation"]},
        ],
    }


class FakeClient:
    responses = {}
    posted = None
    downloads = {}
    deleted = []
    download_limits = []

    def __init__(self, _connection):
        pass

    def get_json(self, path, query=None):
        value = self.responses[path]
        return value(query) if callable(value) else value

    def post_json(self, path, payload):
        self.__class__.posted = (path, payload)
        value = self.responses[path]
        return value(payload) if callable(value) else value

    def delete_json(self, path):
        self.__class__.deleted.append(path)
        return {"id": path.rsplit("/", 1)[-1], "status": "cancelled"}

    def download_bytes(self, url, *, max_bytes=None):
        self.__class__.download_limits.append(max_bytes)
        return self.downloads[url]


@pytest.fixture
def fake_client(monkeypatch):
    FakeClient.responses = {}
    FakeClient.posted = None
    FakeClient.downloads = {}
    FakeClient.deleted = []
    FakeClient.download_limits = []
    monkeypatch.setattr(nodes, "WerkClient", FakeClient)
    return FakeClient


def test_models_parsing_distinguishes_installed_declared_and_available():
    result = classify_image_models(
        models_payload("image-ready", "image-unavailable", "text"), capabilities_payload()
    )
    assert result["installed"] == ["image-ready", "image-unavailable", "text"]
    assert result["declared"] == ["image-ready", "image-unavailable"]
    assert result["available"] == ["image-ready"]


def test_older_models_payload_tasks_are_used_without_capabilities():
    payload = {"data": [{"id": "legacy", "tasks": ["image-generation"], "available_tasks": ["image-generation"]}]}
    result = classify_image_models(payload, {})
    assert result["declared"] == ["legacy"]
    assert result["available"] == ["legacy"]


def test_model_task_normalization_accepts_json_and_display_spellings():
    payload = {
        "models": [
            {
                "id": "mixed",
                "tasks": [" image_generation ", "image-generation", None, 1],
                "available_tasks": ["IMAGE_GENERATION", "image-generation"],
            }
        ]
    }
    result = classify_image_models(models_payload("mixed"), payload)
    assert result["declared"] == ["mixed"]
    assert result["available"] == ["mixed"]
    assert result["models"][0]["tasks"] == ["image-generation"]
    assert result["models"][0]["available_tasks"] == ["image-generation"]


def test_video_models_are_classified_per_generation_task_and_as_a_union():
    capabilities = {
        "models": [
            {
                "id": "text-video",
                "tasks": ["video_generation"],
                "available_tasks": ["video_generation"],
            },
            {
                "id": "wan-ti2v",
                "tasks": ["image-to-video"],
                "available_tasks": [],
            },
            {
                "id": "both",
                "tasks": ["video_generation", "image_to_video"],
                "available_tasks": ["VIDEO_GENERATION", "IMAGE_TO_VIDEO"],
            },
        ]
    }
    result = classify_video_models(
        models_payload("text-video", "wan-ti2v", "both", "text"), capabilities
    )
    assert result["declared"] == ["text-video", "wan-ti2v", "both"]
    assert result["available"] == ["text-video", "both"]
    assert result["by_task"]["video-generation"] == {
        "declared": ["text-video", "both"],
        "available": ["text-video", "both"],
    }
    assert result["by_task"]["image-to-video"] == {
        "declared": ["wan-ti2v", "both"],
        "available": ["both"],
    }
    assert result["models"][1]["declares_image_to_video"] is True
    assert result["models"][1]["image_to_video_probe_eligible"] is False


def test_routing_config_exposes_all_current_werk_request_options():
    config = build_routing_config(
        backend="diffusers",
        accelerator="cuda",
        device="cuda:0",
        precision="bf16",
        quantization="int8",
        profile="flux-memory",
        quality="maximum",
        performance_preference="memory",
        fallback_policy="degrade",
        parameter_policy="warn",
        allow_cpu_offload="enabled",
        allow_sequential_offload="disabled",
        allow_component_offload="enabled",
        allow_disk_offload="disabled",
        attention_backend="sdpa",
        compile="enabled",
        inference_timeout_seconds=1800,
    )

    assert dict(config.request_options) == {
        "backend": "diffusers",
        "accelerator": "cuda",
        "device": "cuda:0",
        "precision": "bf16",
        "quantization": "int8",
        "profile": "flux-memory",
        "quality": "maximum",
        "performance_preference": "memory",
        "fallback_policy": "degrade",
        "parameter_policy": "warn",
        "allow_cpu_offload": True,
        "allow_sequential_offload": False,
        "allow_component_offload": True,
        "allow_disk_offload": False,
        "attention_backend": "sdpa",
        "compile": True,
        "timeout_seconds": 1800,
    }
    assert set(config.request_options) == set(nodes.ROUTING_OPTION_PATHS)
    assert dict(config.parameters) == {}

    required = WerkRoutingConfigNode.INPUT_TYPES()["required"]
    expected_widgets = (set(nodes.ROUTING_OPTION_PATHS) - {"timeout_seconds"}) | {
        "inference_timeout_seconds",
        "additional_routing_parameters_json",
    }
    assert set(required) == expected_widgets


def test_routing_config_inherit_and_empty_values_do_not_invent_overrides():
    config = build_routing_config(
        backend="  ",
        accelerator="",
        quality="inherit",
        performance_preference="inherit",
        fallback_policy="inherit",
        parameter_policy="inherit",
        allow_cpu_offload="inherit",
        allow_sequential_offload="inherit",
        allow_component_offload="inherit",
        allow_disk_offload="inherit",
        compile="inherit",
        inference_timeout_seconds=0,
    )
    assert dict(config.request_options) == {}
    assert dict(config.parameters) == {}


def test_connection_and_inference_timeouts_have_no_artificial_ui_maximum():
    connection_options = WerkConnectionNode.INPUT_TYPES()["required"][
        "timeout_seconds"
    ][1]
    inference_options = WerkRoutingConfigNode.INPUT_TYPES()["required"][
        "inference_timeout_seconds"
    ][1]
    assert connection_options["min"] == 1
    assert "max" not in connection_options
    assert inference_options["min"] == 0
    assert "max" not in inference_options

    connection, _status = WerkConnectionNode().connect(
        "http://werk", "", 86_401, True
    )
    assert connection.timeout_seconds == 86_401

    routing = build_routing_config(inference_timeout_seconds=0x1_0000_0000)
    assert routing.request_options["timeout_seconds"] == 0x1_0000_0000
    with pytest.raises(ValueError, match="must be at least 0"):
        build_routing_config(inference_timeout_seconds=-1)


@pytest.mark.parametrize(
    "field",
    [
        "allow_cpu_offload",
        "allow_sequential_offload",
        "allow_component_offload",
        "allow_disk_offload",
        "compile",
    ],
)
def test_routing_tristate_preserves_explicit_true_false_and_inherit(field):
    assert dict(build_routing_config(**{field: "enabled"}).request_options)[field] is True
    assert dict(build_routing_config(**{field: "disabled"}).request_options)[field] is False
    assert field not in build_routing_config(**{field: "inherit"}).request_options
    with pytest.raises(ValueError, match="inherit, enabled, or disabled"):
        build_routing_config(**{field: "sometimes"})


def test_routing_json_normalizes_bare_and_nested_namespaces():
    normalized = normalize_routing_config_parameters(
        '{"experimental_memory_guard":true,"routing":{"queue_priority":3}}'
    )
    assert normalized == {
        "routing.experimental_memory_guard": True,
        "routing.queue_priority": 3,
    }


@pytest.mark.parametrize(
    "value, message",
    [
        ('{"image.scheduler":"euler"}', "must use the 'routing.' namespace"),
        ('{"precision":"fp16"}', "duplicates a dedicated"),
        ('{"backend":"diffusers"}', "duplicates a dedicated"),
        ('{"queue_priority":1,"routing":{"queue_priority":2}}', "duplicated after normalization"),
    ],
)
def test_routing_json_rejects_wrong_namespace_and_duplicate_controls(value, message):
    with pytest.raises(ValueError, match=message):
        normalize_routing_config_parameters(value)


def test_image_config_json_normalizes_bare_and_nested_namespaces():
    normalized = normalize_image_config_parameters(
        '{"scheduler":"flow_match","image":{"clip_skip":2}}'
    )
    assert normalized == {
        "image.scheduler": "flow_match",
        "image.clip_skip": 2,
    }


@pytest.mark.parametrize(
    "value, message",
    [
        ('{"routing.allow_cpu_offload":true}', "reserved"),
        ('{"width":2048}', "duplicates a dedicated"),
        ('{"batch_size":2}', "duplicates a dedicated"),
        ('{"vae_tiling":true}', "duplicates a dedicated"),
        ('{"scheduler":"a","image":{"scheduler":"b"}}', "duplicated after normalization"),
    ],
)
def test_image_config_json_rejects_wrong_namespace_and_duplicate_controls(value, message):
    with pytest.raises(ValueError, match=message):
        normalize_image_config_parameters(value)


def test_video_config_json_normalizes_and_rejects_dedicated_or_transport_fields():
    assert normalize_video_config_parameters(
        '{"scheduler":"flow_match","video":{"motion_strength":0.7}}'
    ) == {
        "video.scheduler": "flow_match",
        "video.motion_strength": 0.7,
    }
    for value, message in [
        ('{"frames":49}', "duplicates a dedicated"),
        ('{"temporal_vae_tiling":true}', "duplicates a dedicated"),
        ('{"initial_image":"secret"}', "reserved"),
        ('{"image.scheduler":"euler"}', "must use the 'video.' namespace"),
    ]:
        with pytest.raises(ValueError, match=message):
            normalize_video_config_parameters(value)


def test_typed_config_nodes_return_complete_serializable_payloads():
    routing, routing_json = WerkRoutingConfigNode().configure(
        allow_cpu_offload="enabled",
        quality="high",
        additional_routing_parameters_json='{"queue_priority":2}',
    )
    assert isinstance(routing, WerkRoutingConfig)
    assert json.loads(routing_json) == routing_config_payload(routing)

    config, config_json = WerkImageConfigNode().configure(
        routing=routing,
        width=1280,
        height=768,
        count=2,
        batch_size=1,
        steps=24,
        guidance=3.5,
        seed=123,
        output_format="webp",
        response_format="b64_json",
        style="natural",
        vae_tiling="enabled",
        vae_slicing="disabled",
        additional_image_parameters_json='{"scheduler":"flow_match"}',
    )
    assert isinstance(config, WerkImageConfig)
    assert json.loads(config_json) == image_config_payload(config)
    assert dict(config.request_fields) == {
        "n": 2,
        "size": "1280x768",
        "response_format": "b64_json",
        "output_format": "webp",
        "style": "natural",
    }
    assert dict(config.parameters) == {
        "image.steps": 24,
        "image.guidance": 3.5,
        "image.seed": 123,
        "image.vae_tiling": True,
        "image.vae_slicing": False,
        "image.scheduler": "flow_match",
    }

    video, video_json = WerkVideoConfigNode().configure(
        routing=routing,
        width=832,
        height=480,
        count=2,
        batch_size=1,
        frames=49,
        fps=16.0,
        steps=24,
        guidance=5.5,
        seed=987,
        output_format="mp4",
        temporal_vae_tiling="enabled",
        additional_video_parameters_json='{"motion_strength":0.6}',
    )
    assert isinstance(video, WerkVideoConfig)
    assert json.loads(video_json) == video_config_payload(video)
    assert dict(video.request_fields) == {
        "n": 2,
        "size": "832x480",
        "response_format": "mp4",
    }
    assert dict(video.parameters) == {
        "video.frames": 49,
        "video.fps": 16.0,
        "video.steps": 24,
        "video.guidance": 5.5,
        "video.seed": 987,
        "video.temporal_vae_tiling": True,
        "video.motion_strength": 0.6,
    }


def test_image_count_and_batch_size_are_alternative_explicit_controls():
    default = build_image_config(count=1, batch_size=1)
    assert default.request_fields["n"] == 1
    assert "image.batch_size" not in default.parameters

    batched = build_image_config(count=1, batch_size=4)
    assert "n" not in batched.request_fields
    assert batched.parameters["image.batch_size"] == 4

    with pytest.raises(ValueError, match="cannot both be greater than 1"):
        build_image_config(count=2, batch_size=4)


def test_video_count_and_batch_size_are_alternative_explicit_controls():
    default = build_video_config(count=1, batch_size=1)
    assert default.request_fields["n"] == 1
    assert "video.batch_size" not in default.parameters

    batched = build_video_config(count=1, batch_size=4)
    assert "n" not in batched.request_fields
    assert batched.parameters["video.batch_size"] == 4

    with pytest.raises(ValueError, match="cannot both be greater than 1"):
        build_video_config(count=2, batch_size=4)


def test_image_config_forwards_values_above_the_former_portable_caps():
    config = build_image_config(
        width=32_776,
        height=32_776,
        count=1_025,
        batch_size=1,
        steps=1_001,
        guidance=100.1,
    )
    assert config.request_fields["size"] == "32776x32776"
    assert config.request_fields["n"] == 1_025
    assert config.parameters["image.steps"] == 1_001
    assert config.parameters["image.guidance"] == 100.1

    batched = build_image_config(count=1, batch_size=257)
    assert "n" not in batched.request_fields
    assert batched.parameters["image.batch_size"] == 257


def test_video_config_forwards_values_above_the_former_portable_caps():
    config = build_video_config(
        width=16_392,
        height=16_392,
        count=257,
        batch_size=1,
        frames=100_001,
        fps=1_000.1,
        steps=2_001,
        guidance=100.1,
    )
    assert config.request_fields["size"] == "16392x16392"
    assert config.request_fields["n"] == 257
    assert config.parameters["video.frames"] == 100_001
    assert config.parameters["video.fps"] == 1_000.1
    assert config.parameters["video.steps"] == 2_001
    assert config.parameters["video.guidance"] == 100.1

    batched = build_video_config(count=1, batch_size=65)
    assert "n" not in batched.request_fields
    assert batched.parameters["video.batch_size"] == 65


@pytest.mark.parametrize(
    ("node", "unbounded_fields"),
    (
        (
            WerkImageConfigNode,
            ("width", "height", "count", "batch_size", "steps", "guidance"),
        ),
        (
            WerkVideoConfigNode,
            (
                "width",
                "height",
                "count",
                "batch_size",
                "frames",
                "fps",
                "steps",
                "guidance",
            ),
        ),
    ),
)
def test_image_and_video_widgets_have_no_static_inference_maximum(
    node, unbounded_fields
):
    required = node.INPUT_TYPES()["required"]
    for field in unbounded_fields:
        assert "max" not in required[field][1]
        assert "min" in required[field][1]
    assert required["seed"][1]["max"] == 0x7FFFFFFFFFFFFFFF


def test_image_and_video_configs_keep_minimum_and_finite_checks():
    with pytest.raises(ValueError, match="width and height must be at least 64"):
        build_image_config(width=63)
    with pytest.raises(ValueError, match="steps must be at least 1"):
        build_video_config(steps=0)
    with pytest.raises(ValueError, match="guidance must be finite"):
        build_image_config(guidance=float("nan"))
    with pytest.raises(ValueError, match="fps must be finite"):
        build_video_config(fps=float("inf"))


def test_flux_request_carries_offload_as_explicit_routing_options():
    routing = build_routing_config(
        backend="media-companion",
        accelerator="cuda",
        allow_cpu_offload="enabled",
        allow_sequential_offload="enabled",
        allow_component_offload="disabled",
        performance_preference="memory",
        additional_routing_parameters_json='{"experimental_memory_guard":true}',
    )
    config = build_image_config(
        width=1024,
        height=1024,
        steps=20,
        guidance=3.5,
        seed=99,
        additional_image_parameters_json='{"scheduler":"flow_match"}',
        routing=routing,
    )
    request = build_configured_image_request(
        model="black-forest-labs/FLUX.2-klein-4B",
        prompt="a small red robot in a forest",
        config=config,
    )

    assert request["model"] == "black-forest-labs/FLUX.2-klein-4B"
    assert request["backend"] == "media-companion"
    assert request["accelerator"] == "cuda"
    assert request["allow_cpu_offload"] is True
    assert request["allow_sequential_offload"] is True
    assert request["allow_component_offload"] is False
    assert request["performance_preference"] == "memory"
    assert request["parameters"]["routing.experimental_memory_guard"] is True
    assert request["parameters"]["image.scheduler"] == "flow_match"
    assert "routing" not in request
    assert "routing.allow_cpu_offload" not in request["parameters"]


def test_generator_requires_a_real_linkable_model_input():
    inputs = WerkImageGenerateNode.INPUT_TYPES()
    model_type, model_options = inputs["required"]["model"]
    assert model_type == "STRING"
    assert model_options["forceInput"] is True
    assert "default" not in model_options
    assert inputs["optional"]["config"] == ("WERK_IMAGE_CONFIG",)

    video_inputs = WerkVideoGenerateNode.INPUT_TYPES()
    video_model_type, video_model_options = video_inputs["required"]["model"]
    assert video_model_type == "STRING"
    assert video_model_options["forceInput"] is True
    assert video_inputs["optional"] == {
        "config": ("WERK_VIDEO_CONFIG",),
        "initial_image": ("IMAGE",),
    }
    assert WerkVideoGenerateNode.RETURN_TYPES[0] == "VIDEO"
    assert WerkVideoGenerateNode.OUTPUT_IS_LIST[0] is True


def test_configured_video_request_merges_routing_and_embeds_one_initial_image():
    routing = build_routing_config(
        backend="media-companion",
        allow_cpu_offload="enabled",
        inference_timeout_seconds=1800,
    )
    config = build_video_config(
        width=832,
        height=480,
        count=1,
        frames=49,
        fps=16,
        steps=20,
        guidance=5,
        seed=42,
        temporal_vae_tiling="enabled",
        additional_video_parameters_json='{"flow_shift":3.0}',
        routing=routing,
    )
    image = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    request = build_configured_video_request(
        model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        prompt="slow camera orbit",
        negative_prompt="watermark",
        initial_image=image,
        config=config,
    )
    assert request["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert request["size"] == "832x480"
    assert request["response_format"] == "mp4"
    assert request["backend"] == "media-companion"
    assert request["allow_cpu_offload"] is True
    assert request["timeout_seconds"] == 1800
    assert request["parameters"] == {
        "video.frames": 49,
        "video.fps": 16.0,
        "video.steps": 20,
        "video.guidance": 5.0,
        "video.seed": 42,
        "video.temporal_vae_tiling": True,
        "video.flow_shift": 3.0,
    }
    assert request["initial_image"]["mime_type"] == "image/png"
    with Image.open(BytesIO(base64.b64decode(request["initial_image"]["base64"]))) as encoded:
        assert encoded.size == (1, 1)

    without_image = build_configured_video_request(
        model="text-video", prompt="clouds", config=build_video_config()
    )
    assert "initial_image" not in without_image


def test_server_info_tolerates_optional_capabilities_failure(fake_client):
    fake_client.responses["/v1/models"] = models_payload("legacy")

    def unavailable(_query):
        raise nodes.WerkApiError("GET", "http://werk/v1/capabilities", 404, "not found")

    fake_client.responses["/v1/capabilities"] = unavailable
    models, capabilities, metadata = WerkServerInfoNode().discover(WerkConnection("http://werk"), 0)
    assert models == "legacy"
    assert json.loads(capabilities) == {}
    assert "capabilities_warning" in json.loads(metadata)


def test_image_model_selection_prefers_probe_available(fake_client):
    fake_client.responses = {
        "/v1/models": models_payload("image-ready", "image-unavailable", "text"),
        "/v1/capabilities": capabilities_payload(),
    }
    selected, choices, metadata = WerkImageModelsNode().select(WerkConnection("http://werk"), 0, "", True)
    assert selected == "image-ready"
    assert choices == "image-ready"
    assert json.loads(metadata)["declared"] == ["image-ready", "image-unavailable"]


def test_multiple_image_models_require_explicit_preference(fake_client):
    fake_client.responses = {
        "/v1/models": models_payload("a", "b"),
        "/v1/capabilities": {
            "models": [
                {"id": "a", "tasks": ["image-generation"], "available_tasks": ["image-generation"]},
                {"id": "b", "tasks": ["image_generation"], "available_tasks": ["image_generation"]},
            ]
        },
    }
    with pytest.raises(ValueError, match="a, b"):
        WerkImageModelsNode().select(WerkConnection("http://werk"), 0, "", True)


def test_video_model_selection_filters_the_requested_task(fake_client):
    fake_client.responses = {
        "/v1/models": models_payload("text-video", "wan-ti2v"),
        "/v1/capabilities": {
            "models": [
                {
                    "id": "text-video",
                    "tasks": ["video_generation"],
                    "available_tasks": ["video_generation"],
                },
                {
                    "id": "wan-ti2v",
                    "tasks": ["image_to_video"],
                    "available_tasks": ["image_to_video"],
                },
            ]
        },
    }
    selected, choices, metadata = WerkVideoModelsNode().select(
        WerkConnection("http://werk"), "image_to_video", 0, "", True
    )
    assert selected == "wan-ti2v"
    assert choices == "wan-ti2v"
    assert json.loads(metadata)["selected_task"] == "image-to-video"

    with pytest.raises(ValueError, match="task must be"):
        WerkVideoModelsNode().select(
            WerkConnection("http://werk"), "video-upscaling", 0, "", True
        )


def test_parameter_schema_query_is_forwarded_and_returned(fake_client):
    seen = {}

    def response(query):
        seen.update(query)
        return {"parameters": [{"path": "image.steps", "default": 28, "minimum": 1}]}

    fake_client.responses["/v1/parameters"] = response
    payload, summary = WerkImageParametersNode().parameters(WerkConnection("http://werk"), "tiny-sd", "auto", 3)
    assert seen == {"task": "image-generation", "model": "tiny-sd", "backend": "auto"}
    assert json.loads(payload)["parameters"][0]["path"] == "image.steps"
    assert "1 parameter descriptor" in summary


def test_video_parameter_schema_uses_the_selected_task(fake_client):
    seen = {}

    def response(query):
        seen.update(query)
        return {"parameters": [{"path": "video.frames", "default": 81}]}

    fake_client.responses["/v1/parameters"] = response
    payload, summary = WerkVideoParametersNode().parameters(
        WerkConnection("http://werk"),
        "wan-ti2v",
        "image-to-video",
        "diffusers",
        1,
    )
    assert seen == {
        "task": "image-to-video",
        "model": "wan-ti2v",
        "backend": "diffusers",
    }
    assert json.loads(payload)["parameters"][0]["path"] == "video.frames"
    assert "image-to-video" in summary


def response_with_images(entries):
    return {
        "created": 1,
        "data": entries,
        "werk": {
            "id": "result-1",
            "task": "image-generation",
            "model": "tiny-sd",
            "runtime": "test",
            "effective_request": {},
            "estimate": {},
            "plan": {},
            "backend_metadata": {"path": "/private/model"},
            "timings": {},
            "warnings": [],
            "created_unix": 1,
            "outputs": [{"path": "/private/output.png"}],
        },
    }


def test_generate_handles_base64_batch_ids_and_sanitized_metadata(fake_client):
    encoded = base64.b64encode(image_bytes()).decode()
    fake_client.responses["/v1/images/generations"] = response_with_images(
        [
            {"id": "out-1", "b64_json": encoded, "mime_type": "image/png"},
            {"id": "out-2", "b64_json": encoded, "mime_type": "image/png"},
        ]
    )
    config = build_image_config(
        width=512,
        height=768,
        count=2,
        steps=20,
        guidance=6.5,
        seed=42,
    )
    image, metadata, seed, result_id, output_ids = WerkImageGenerateNode().generate(
        WerkConnection("http://werk"),
        model="tiny-sd",
        prompt="robot",
        negative_prompt="bad geometry",
        config=config,
    )
    assert image.shape == (2, 2, 2, 3)
    assert seed == 42
    assert result_id == "result-1"
    assert output_ids == "out-1\nout-2"
    assert "/private" not in metadata
    assert fake_client.posted[1]["parameters"]["image.guidance"] == 6.5


def test_generate_handles_relative_url_output(fake_client):
    fake_client.responses["/v1/images/generations"] = response_with_images(
        [{"id": "out-url", "url": "/v1/outputs/out-url", "mime_type": "image/png"}]
    )
    fake_client.downloads["/v1/outputs/out-url"] = (image_bytes(), "image/png")
    config = build_image_config(response_format="url", seed=42)
    image, _metadata, _seed, result_id, output_ids = WerkImageGenerateNode().generate(
        WerkConnection("http://werk"),
        model="tiny-sd",
        prompt="robot",
        negative_prompt="",
        config=config,
    )
    assert image.shape == (1, 2, 2, 3)
    assert result_id == "result-1"
    assert output_ids == "out-url"


def test_generate_executes_typed_config_and_posts_merged_request(fake_client):
    encoded = base64.b64encode(image_bytes((3, 2))).decode()
    fake_client.responses["/v1/images/generations"] = response_with_images(
        [{"id": "flux-output", "b64_json": encoded, "mime_type": "image/png"}]
    )
    routing = build_routing_config(
        backend="diffusers",
        allow_cpu_offload="enabled",
        quality="high",
        inference_timeout_seconds=1200,
    )
    config = build_image_config(
        width=768,
        height=512,
        count=1,
        batch_size=1,
        steps=18,
        guidance=3.2,
        seed=456,
        output_format="png",
        response_format="b64_json",
        style="natural",
        vae_tiling="enabled",
        additional_image_parameters_json='{"scheduler":"flow_match"}',
        routing=routing,
    )

    image, metadata, seed, result_id, output_ids = WerkImageGenerateNode().generate(
        WerkConnection("http://werk"),
        model="black-forest-labs/FLUX.2-klein-4B",
        prompt="a detailed red robot",
        negative_prompt="watermark",
        config=config,
    )

    assert image.shape == (1, 2, 3, 3)
    assert seed == 456
    assert result_id == "result-1"
    assert output_ids == "flux-output"
    assert "/private" not in metadata
    path, payload = fake_client.posted
    assert path == "/v1/images/generations"
    assert payload["model"] == "black-forest-labs/FLUX.2-klein-4B"
    assert payload["prompt"] == "a detailed red robot"
    assert payload["negative_prompt"] == "watermark"
    assert payload["size"] == "768x512"
    assert payload["style"] == "natural"
    assert payload["backend"] == "diffusers"
    assert payload["allow_cpu_offload"] is True
    assert payload["quality"] == "high"
    assert payload["timeout_seconds"] == 1200
    assert payload["parameters"] == {
        "image.steps": 18,
        "image.guidance": 3.2,
        "image.seed": 456,
        "image.vae_tiling": True,
        "image.scheduler": "flow_match",
    }


def completed_video_job():
    return {
        "id": "job-video-1",
        "status": "completed",
        "created_unix": 1,
        "updated_unix": 2,
        "request": {
            "inputs": [
                {"source": {"kind": "base64", "data": "request-secret"}}
            ]
        },
        "result": {
            "id": "result-video-1",
            "task": "image_to_video",
            "model": "wan-ti2v",
            "runtime": "media-companion-cuda",
            "effective_request": {
                "inputs": [
                    {
                        "modality": "image",
                        "role": "initial_image",
                        "source": {
                            "kind": "base64",
                            "data": "effective-secret",
                        },
                    }
                ]
            },
            "estimate": {},
            "plan": {},
            "backend_metadata": {"path": "/private/model"},
            "timings": {"total_seconds": 1.2},
            "warnings": [],
            "created_unix": 2,
            "outputs": [
                {
                    "id": "video-output-1",
                    "path": "/private/output.mp4",
                    "mime_type": "video/mp4",
                    "size_bytes": 13,
                    "width": 832,
                    "height": 480,
                    "duration": 3.0,
                    "seed": 42,
                }
            ],
        },
        "error": None,
    }


def test_video_generate_posts_polls_downloads_and_returns_native_video_list(
    fake_client, monkeypatch
):
    fake_client.responses["/v1/videos/generations"] = {
        "id": "job-video-1",
        "status": "queued",
        "created_unix": 1,
        "updated_unix": 1,
    }
    fake_client.responses["/v1/jobs/job-video-1"] = completed_video_job()
    fake_client.downloads["/v1/outputs/video-output-1"] = (
        b"encoded-video",
        "video/mp4",
    )
    monkeypatch.setattr(nodes.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        nodes,
        "video_bytes_to_comfy",
        lambda raw: {"native_video": raw},
    )

    videos, metadata, seed, job_id, result_id, output_ids = (
        WerkVideoGenerateNode().generate(
            WerkConnection("http://werk", timeout_seconds=30),
            model="wan-ti2v",
            prompt="slow orbit",
            negative_prompt="watermark",
            config=build_video_config(seed=42),
        )
    )
    assert videos == [{"native_video": b"encoded-video"}]
    assert seed == 42
    assert job_id == "job-video-1"
    assert result_id == "result-video-1"
    assert output_ids == "video-output-1"
    assert fake_client.posted[0] == "/v1/videos/generations"
    assert fake_client.download_limits == [nodes.environment_max_video_bytes()]
    assert "/private" not in metadata
    assert "request-secret" not in metadata
    assert "effective-secret" not in metadata
    assert json.loads(metadata)["result"]["effective_request"]["inputs"][0][
        "source"
    ] == {"kind": "base64", "embedded": True}


def test_video_job_timeout_cancels_best_effort(fake_client):
    moments = iter([0.0, 2.0])
    client = fake_client(WerkConnection("http://werk"))
    with pytest.raises(TimeoutError, match="within 1 seconds"):
        wait_for_video_job(
            client,
            {"id": "job-timeout", "status": "running"},
            timeout_seconds=1,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: next(moments),
            interrupt_check=lambda: None,
        )
    assert fake_client.deleted == ["/v1/jobs/job-timeout"]


def test_video_job_comfy_interrupt_cancels_best_effort(fake_client):
    client = fake_client(WerkConnection("http://werk"))

    def interrupted():
        raise RuntimeError("ComfyUI processing interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        wait_for_video_job(
            client,
            {"id": "job-interrupted", "status": "running"},
            timeout_seconds=30,
            interrupt_check=interrupted,
        )
    assert fake_client.deleted == ["/v1/jobs/job-interrupted"]


def test_video_job_failure_is_terminal_and_actionable(fake_client):
    client = fake_client(WerkConnection("http://werk"))
    with pytest.raises(ValueError, match="encoder exploded"):
        wait_for_video_job(
            client,
            {"id": "job-failed", "status": "failed", "error": "encoder exploded"},
            timeout_seconds=1,
        )
    assert fake_client.deleted == []


def test_configured_request_rejects_parameter_collisions_instead_of_overwriting():
    routing = WerkRoutingConfig(parameters={"image.scheduler": "routing-value"})
    config = WerkImageConfig(
        parameters={"image.scheduler": "image-value"},
        routing=routing,
    )
    with pytest.raises(ValueError, match="config parameter collision: image.scheduler"):
        build_configured_image_request(model="tiny-sd", prompt="robot", config=config)

    video_routing = WerkRoutingConfig(
        parameters={"video.scheduler": "routing-value"}
    )
    video_config = WerkVideoConfig(
        parameters={"video.scheduler": "video-value"},
        routing=video_routing,
    )
    with pytest.raises(ValueError, match="config parameter collision: video.scheduler"):
        build_configured_video_request(
            model="wan", prompt="robot", config=video_config
        )


def test_node_exports_and_display_names_are_complete():
    expected = {
        "WerkConnection",
        "WerkServerInfo",
        "WerkImageModels",
        "WerkImageParameters",
        "WerkRoutingConfig",
        "WerkImageConfig",
        "WerkImageGenerate",
        "WerkVideoModels",
        "WerkVideoParameters",
        "WerkVideoConfig",
        "WerkVideoGenerate",
        "WerkAudioModels",
        "WerkAudioParameters",
        "WerkAudioConfig",
        "WerkAudioGenerate",
        "WerkAudioProcess",
        "WerkAudioAnalyze",
    }
    assert set(NODE_CLASS_MAPPINGS) == expected
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == expected
    assert NODE_CLASS_MAPPINGS["WerkRoutingConfig"] is WerkRoutingConfigNode
    assert NODE_CLASS_MAPPINGS["WerkImageConfig"] is WerkImageConfigNode
    assert NODE_CLASS_MAPPINGS["WerkImageGenerate"] is WerkImageGenerateNode
    assert NODE_CLASS_MAPPINGS["WerkVideoModels"] is WerkVideoModelsNode
    assert NODE_CLASS_MAPPINGS["WerkVideoParameters"] is WerkVideoParametersNode
    assert NODE_CLASS_MAPPINGS["WerkVideoConfig"] is WerkVideoConfigNode
    assert NODE_CLASS_MAPPINGS["WerkVideoGenerate"] is WerkVideoGenerateNode
    assert NODE_DISPLAY_NAME_MAPPINGS["WerkImageGenerate"] == "WERK Image Generate (Beta)"
    assert NODE_DISPLAY_NAME_MAPPINGS["WerkVideoGenerate"] == "WERK Video Generate (Beta)"
    assert all(name.endswith(" (Beta)") for name in NODE_DISPLAY_NAME_MAPPINGS.values())
    assert WEB_DIRECTORY == "./web/js"
    assert (Path(__file__).parents[1] / "web" / "js" / "werk_ui.js").is_file()
    frontend = (Path(__file__).parents[1] / "web" / "js" / "werk_ui.js").read_text()
    assert 'const VIDEO_MODELS_CLASS = "WerkVideoModels"' in frontend
    assert "updateVideoModelsNode" in frontend
    assert 'const AUDIO_MODELS_CLASS = "WerkAudioModels"' in frontend
    assert "updateAudioModelsNode" in frontend


def test_example_workflows_are_distinct_valid_json_shapes():
    examples = Path(__file__).parents[1] / "examples"
    ui = json.loads((examples / "werk_image_generation_workflow.json").read_text())
    api = json.loads((examples / "werk_image_generation_api.json").read_text())
    video_api = json.loads((examples / "werk_video_generation_api.json").read_text())
    image_to_video_api = json.loads(
        (examples / "werk_image_to_video_api.json").read_text()
    )
    assert ui["version"] == 0.4
    assert {node["type"] for node in ui["nodes"]} >= {
        "WerkConnection",
        "WerkImageModels",
        "WerkRoutingConfig",
        "WerkImageConfig",
        "WerkImageGenerate",
        "PreviewImage",
        "SaveImage",
    }
    node_ids = {str(node["id"]) for node in ui["nodes"]}
    assert all(str(link[1]) in node_ids and str(link[3]) in node_ids for link in ui["links"])
    assert api["2"]["class_type"] == "WerkImageModels"
    assert api["2"]["inputs"]["connection"] == ["1", 0]
    assert api["3"]["class_type"] == "WerkRoutingConfig"
    assert api["4"]["class_type"] == "WerkImageConfig"
    assert api["4"]["inputs"]["routing"] == ["3", 0]
    assert api["5"]["class_type"] == "WerkImageGenerate"
    assert api["5"]["inputs"]["connection"] == ["1", 0]
    assert api["5"]["inputs"]["model"] == ["2", 0]
    assert api["5"]["inputs"]["config"] == ["4", 0]
    assert api["6"]["inputs"]["images"] == ["5", 0]
    assert api["7"]["inputs"]["images"] == ["5", 0]

    assert video_api["2"]["class_type"] == "WerkVideoModels"
    assert video_api["2"]["inputs"]["task"] == "video-generation"
    assert video_api["3"]["inputs"]["precision"] == "bf16"
    assert video_api["4"]["class_type"] == "WerkVideoConfig"
    assert video_api["5"]["class_type"] == "WerkVideoGenerate"
    assert "initial_image" not in video_api["5"]["inputs"]
    assert video_api["6"]["inputs"]["video"] == ["5", 0]

    assert image_to_video_api["2"]["inputs"]["task"] == "image-to-video"
    assert image_to_video_api["3"]["inputs"]["precision"] == "bf16"
    assert image_to_video_api["5"]["class_type"] == "LoadImage"
    assert image_to_video_api["6"]["class_type"] == "WerkVideoGenerate"
    assert image_to_video_api["6"]["inputs"]["initial_image"] == ["5", 0]
    assert image_to_video_api["7"]["inputs"]["video"] == ["6", 0]


def test_package_reload_performs_no_network_request(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network request during import")

    monkeypatch.setattr(nodes.WerkClient, "get_json", forbidden)
    importlib.reload(nodes)
