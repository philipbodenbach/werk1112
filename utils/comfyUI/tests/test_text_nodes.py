import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, text_nodes
from ..config import WerkConnection, WerkTextConfig
from ..text_nodes import (
    WerkTextConfigNode,
    WerkTextGenerateNode,
    WerkTextModelsNode,
    build_text_config,
    build_text_request,
    classify_text_models,
    execute_text_request,
    text_config_payload,
)
from .test_protocol import envelope, send, servers


def completion():
    return {
        "id": "chatcmpl-text-test",
        "object": "chat.completion",
        "model": "deepseek-v4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello", "reasoning_content": "private reasoning"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "echoed_request": "must not enter metadata",
    }


def capability(status="supported", operations=None):
    return {"capabilities": [{
        "id": "api.chat.omlx_options", "status": status, "detail": "request controls",
        "operations": ["thinking", "expert_cache_mb"] if operations is None else operations,
    }]}


def test_inherited_config_omits_extension_and_preserves_standard_chat_contract():
    request = build_text_request(model=" model-id ", prompt="Hello")
    assert request == {
        "model": "model-id", "messages": [{"role": "user", "content": "Hello"}],
        "stream": False, "temperature": 0.2, "top_p": 1.0,
        "max_completion_tokens": 1024, "seed": 0,
    }
    assert "werk" not in text_config_payload(build_text_config(omlx_expert_cache_mb=1))
    assert "routing" not in request


@pytest.mark.parametrize("thinking,offload,budget,expected", [
    ("disabled", "enabled", 8192, {"thinking": False, "expert_cache_mb": 8192}),
    ("enabled", "disabled", 8192, {"thinking": True, "expert_cache_mb": 0}),
    ("disabled", "disabled", 8192, {"thinking": False, "expert_cache_mb": 0}),
    ("inherit", "enabled", 1, {"expert_cache_mb": 1}),
    ("inherit", "enabled", 1048576, {"expert_cache_mb": 1048576}),
    ("enabled", "inherit", 8192, {"thinking": True}),
])
def test_explicit_options_preserve_false_zero_and_budget_boundaries(thinking, offload, budget, expected):
    config = build_text_config(
        omlx_thinking=thinking, omlx_expert_offload=offload, omlx_expert_cache_mb=budget,
    )
    assert text_config_payload(config)["werk"] == {"omlx": expected}


@pytest.mark.parametrize("budget", [0, -1, 1048577, True, 1.5, "8192"])
def test_enabled_expert_cache_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="integer from 1 to 1048576"):
        build_text_config(omlx_expert_offload="enabled", omlx_expert_cache_mb=budget)


@pytest.mark.parametrize("kwargs", [
    {"omlx_thinking": "auto"}, {"omlx_expert_offload": "yes"},
    {"temperature": float("nan")}, {"top_p": 2}, {"max_completion_tokens": 0},
])
def test_invalid_controls_fail_before_inference(kwargs):
    with pytest.raises(ValueError):
        build_text_config(**kwargs)


def test_frozen_text_config_and_payload_copies_cannot_change_other_workflow_branches():
    fields = {"stop": ["END"]}
    options = {"thinking": False, "expert_cache_mb": 8192}
    config = WerkTextConfig(request_fields=fields, omlx_options=options)
    fields["stop"].append("MUTATED")
    options["thinking"] = True
    with pytest.raises(FrozenInstanceError):
        config.omlx_options = {}
    with pytest.raises(TypeError):
        config.omlx_options["thinking"] = True
    payload = text_config_payload(config)
    payload["werk"]["omlx"]["thinking"] = True
    payload["stop"].append("OTHER")
    assert text_config_payload(config) == {
        "stop": ["END"], "werk": {"omlx": {"thinking": False, "expert_cache_mb": 8192}},
    }
    with pytest.raises(ValueError, match="unsupported chat"):
        text_config_payload(WerkTextConfig(request_fields={"stream": True}))
    with pytest.raises(TypeError, match="WerkTextConfig"):
        build_text_request(model="test", prompt="Hi", config={})


def test_messages_preserve_history_order_and_append_current_prompt():
    prior = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Answer"}]
    request = build_text_request(
        model="test", prompt="Next", system_prompt="Be concise", messages_json=json.dumps(prior),
    )
    assert request["messages"] == [
        {"role": "system", "content": "Be concise"}, *prior,
        {"role": "user", "content": "Next"},
    ]


@pytest.mark.parametrize("history", [
    "{}", "not JSON", '[{"role":"tool","content":"bad"}]',
    '[{"role":"user","content":[{"type":"image_url"}]}]',
    '[{"role":[],"content":"bad"}]', '[{"role":"user","content":"x","extra":true}]',
])
def test_text_history_rejects_nontext_or_malformed_messages(history):
    with pytest.raises(ValueError, match="messages_json"):
        build_text_request(model="test", prompt="Hi", messages_json=history)


def test_explicit_options_preflight_versioned_capability_then_send_exact_http_payload(servers):
    def responder(handler, _server):
        if handler.path == "/werk/v1/capabilities":
            send(handler, payload=envelope(capability()))
        else:
            assert handler.path == "/v1/chat/completions"
            send(handler, payload=completion())

    server = servers(responder)
    connection = WerkConnection(server.url, "fixture-token")
    config = build_text_config(omlx_thinking="disabled", omlx_expert_offload="disabled")
    result = WerkTextGenerateNode().generate(connection, "deepseek-v4", "Hi", config=config)
    assert [entry[0] for entry in server.requests] == ["/werk/v1/capabilities", "/v1/chat/completions"]
    assert all(entry[1]["Authorization"] == "Bearer fixture-token" for entry in server.requests)
    assert server.requests[0][1]["X-Werk-Protocol-Version"] == "1.0"
    payload = json.loads(server.requests[1][2])
    assert payload["werk"] == {"omlx": {"thinking": False, "expert_cache_mb": 0}}
    assert payload["stream"] is False
    assert result["ui"] == {"text": ["Hello"]}
    assert result["result"][0] == "Hello"
    assert result["result"][2:] == ("chatcmpl-text-test", "stop", "deepseek-v4")
    assert "private reasoning" not in result["result"][1]
    assert "echoed_request" not in result["result"][1]


@pytest.mark.parametrize("payload", [
    {"capabilities": []}, capability(status="unsupported"),
    capability(status="experimental"), capability(operations=["thinking"]),
])
def test_missing_or_incomplete_capability_blocks_explicit_post(servers, payload):
    def responder(handler, _server):
        send(handler, payload=envelope(payload))

    server = servers(responder)
    request = build_text_request(model="test", prompt="Hi", config=build_text_config(omlx_expert_offload="disabled"))
    with pytest.raises(ValueError, match="Update and restart"):
        execute_text_request(WerkConnection(server.url), request)
    assert [entry[0] for entry in server.requests] == ["/werk/v1/capabilities"]


def test_old_server_without_protocol_fails_explicit_options_without_posting(servers):
    server = servers(lambda handler, _server: send(handler, status=404, payload={"error": "not found"}))
    request = build_text_request(model="test", prompt="Hi", config=build_text_config(omlx_thinking="disabled"))
    with pytest.raises(ValueError, match="Update and restart"):
        execute_text_request(WerkConnection(server.url), request)
    assert len(server.requests) == 1


def test_inherited_defaults_generate_without_protocol_probe(servers):
    server = servers(lambda handler, _server: send(handler, payload=completion()))
    execute_text_request(WerkConnection(server.url), build_text_request(model="test", prompt="Hi"))
    assert [entry[0] for entry in server.requests] == ["/v1/chat/completions"]
    assert "werk" not in json.loads(server.requests[0][2])


def test_text_discovery_respects_declared_and_available_tasks(monkeypatch):
    assert WerkTextModelsNode.INPUT_TYPES()["required"]["require_available"][1]["default"] is False
    models = {"data": [{"id": "text-ready"}, {"id": "text-cold"}, {"id": "image-only"}]}
    capabilities = {"models": [
        {"id": "text-ready", "tasks": ["text_generation"], "available_tasks": ["text_generation"]},
        {"id": "text-cold", "tasks": ["text-generation"], "available_tasks": []},
        {"id": "image-only", "tasks": ["image-generation"], "available_tasks": ["image-generation"]},
    ]}
    classified = classify_text_models(models, capabilities)
    assert classified["declared"] == ["text-ready", "text-cold"]
    assert classified["available"] == ["text-ready"]

    class Client:
        def __init__(self, _connection):
            pass

        def get_json(self, path):
            return models if path == "/v1/models" else capabilities

    monkeypatch.setattr(text_nodes, "WerkClient", Client)
    node = WerkTextModelsNode()
    assert node.select(WerkConnection("http://werk"), 0, "", True)[0] == "text-ready"
    with pytest.raises(ValueError, match="probe-eligible"):
        node.select(WerkConnection("http://werk"), 0, "text-cold", True)
    assert node.select(WerkConnection("http://werk"), 0, "text-cold", False)[0] == "text-cold"


def test_text_nodes_are_exported_and_api_example_is_executable():
    expected = {"WerkTextModels": WerkTextModelsNode, "WerkTextConfig": WerkTextConfigNode, "WerkTextGenerate": WerkTextGenerateNode}
    assert all(NODE_CLASS_MAPPINGS[name] is node for name, node in expected.items())
    assert all(NODE_DISPLAY_NAME_MAPPINGS[name].endswith(" (Beta)") for name in expected)
    assert WerkTextGenerateNode.INPUT_TYPES()["optional"]["config"] == ("WERK_TEXT_CONFIG",)
    assert "images" not in WerkTextGenerateNode.INPUT_TYPES()["required"]
    assert WerkTextGenerateNode.RETURN_NAMES[-1] == "model_id"
    path = Path(__file__).parents[1] / "examples/werk_text_omlx_api.json"
    prompt = json.loads(path.read_text())
    for node in prompt.values():
        assert node["class_type"] in NODE_CLASS_MAPPINGS
        required = NODE_CLASS_MAPPINGS[node["class_type"]].INPUT_TYPES()["required"]
        assert set(required) <= set(node["inputs"])
    config_inputs = next(node["inputs"] for node in prompt.values() if node["class_type"] == "WerkTextConfig")
    assert text_config_payload(build_text_config(**config_inputs))["werk"]["omlx"] == {"thinking": False, "expert_cache_mb": 8192}
    discovery_inputs = next(node["inputs"] for node in prompt.values() if node["class_type"] == "WerkTextModels")
    assert discovery_inputs["require_available"] is False
