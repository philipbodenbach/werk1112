import base64
import importlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
from ..config import WerkConnection, WerkImageConfig, WerkRoutingConfig
from .. import nodes
from ..nodes import (
    WerkImageConfigNode,
    WerkImageGenerateNode,
    WerkImageModelsNode,
    WerkImageParametersNode,
    WerkRoutingConfigNode,
    WerkServerInfoNode,
    build_configured_image_request,
    build_image_config,
    build_routing_config,
    classify_image_models,
    image_config_payload,
    normalize_image_config_parameters,
    normalize_routing_config_parameters,
    routing_config_payload,
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

    def __init__(self, _connection):
        pass

    def get_json(self, path, query=None):
        value = self.responses[path]
        return value(query) if callable(value) else value

    def post_json(self, path, payload):
        self.__class__.posted = (path, payload)
        value = self.responses[path]
        return value(payload) if callable(value) else value

    def download_bytes(self, url):
        return self.downloads[url]


@pytest.fixture
def fake_client(monkeypatch):
    FakeClient.responses = {}
    FakeClient.posted = None
    FakeClient.downloads = {}
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


def test_image_count_and_batch_size_are_alternative_explicit_controls():
    default = build_image_config(count=1, batch_size=1)
    assert default.request_fields["n"] == 1
    assert "image.batch_size" not in default.parameters

    batched = build_image_config(count=1, batch_size=4)
    assert "n" not in batched.request_fields
    assert batched.parameters["image.batch_size"] == 4

    with pytest.raises(ValueError, match="cannot both be greater than 1"):
        build_image_config(count=2, batch_size=4)


def test_flux_request_carries_offload_as_explicit_routing_options():
    routing = build_routing_config(
        backend="diffusers",
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
    assert request["backend"] == "diffusers"
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


def test_configured_request_rejects_parameter_collisions_instead_of_overwriting():
    routing = WerkRoutingConfig(parameters={"image.scheduler": "routing-value"})
    config = WerkImageConfig(
        parameters={"image.scheduler": "image-value"},
        routing=routing,
    )
    with pytest.raises(ValueError, match="config parameter collision: image.scheduler"):
        build_configured_image_request(model="tiny-sd", prompt="robot", config=config)


def test_node_exports_and_display_names_are_complete():
    expected = {
        "WerkConnection",
        "WerkServerInfo",
        "WerkImageModels",
        "WerkImageParameters",
        "WerkRoutingConfig",
        "WerkImageConfig",
        "WerkImageGenerate",
    }
    assert set(NODE_CLASS_MAPPINGS) == expected
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == expected
    assert NODE_CLASS_MAPPINGS["WerkRoutingConfig"] is WerkRoutingConfigNode
    assert NODE_CLASS_MAPPINGS["WerkImageConfig"] is WerkImageConfigNode
    assert NODE_CLASS_MAPPINGS["WerkImageGenerate"] is WerkImageGenerateNode
    assert NODE_DISPLAY_NAME_MAPPINGS["WerkImageGenerate"] == "WERK Image Generate"
    assert WEB_DIRECTORY == "./web/js"
    assert (Path(__file__).parents[1] / "web" / "js" / "werk_ui.js").is_file()


def test_example_workflows_are_distinct_valid_json_shapes():
    examples = Path(__file__).parents[1] / "examples"
    ui = json.loads((examples / "werk_image_generation_workflow.json").read_text())
    api = json.loads((examples / "werk_image_generation_api.json").read_text())
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


def test_package_reload_performs_no_network_request(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network request during import")

    monkeypatch.setattr(nodes.WerkClient, "get_json", forbidden)
    importlib.reload(nodes)
