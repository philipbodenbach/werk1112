import json

import pytest
import torch

from .. import nodes
from ..config import (
    WerkConnection,
    WerkVisionConfig,
    environment_max_vision_input_bytes,
)
from ..nodes import (
    WerkVisionAnalyzeNode,
    WerkVisionConfigNode,
    WerkVisionModelsNode,
    build_vision_config,
    build_vision_request,
    classify_vision_models,
    execute_vision_request,
    vision_config_payload,
)


class FakeClient:
    responses = {}
    posted = None

    def __init__(self, _connection):
        pass

    def get_json(self, path, query=None):
        del query
        return self.responses[path]

    def post_json(self, path, payload):
        self.__class__.posted = (path, payload)
        return self.responses[path]


@pytest.fixture
def fake_client(monkeypatch):
    FakeClient.responses = {}
    FakeClient.posted = None
    monkeypatch.setattr(nodes, "WerkClient", FakeClient)
    return FakeClient


def test_vision_classifier_uses_authoritative_task_and_readiness():
    result = classify_vision_models(
        {
            "data": [
                {"id": "qwen-vl"},
                {"id": "glm-v"},
                {"id": "qwen-text"},
            ]
        },
        {
            "models": [
                {
                    "id": "qwen-vl",
                    "tasks": ["text_generation", "image_understanding"],
                    "available_tasks": ["text_generation", "image_understanding"],
                },
                {
                    "id": "glm-v",
                    "tasks": ["image-understanding"],
                    "available_tasks": [],
                },
                {
                    "id": "qwen-text",
                    "tasks": ["text-generation"],
                    "available_tasks": ["text-generation"],
                },
            ]
        },
    )
    assert result["declared"] == ["qwen-vl", "glm-v"]
    assert result["available"] == ["qwen-vl"]
    assert result["models"][0]["image_understanding_probe_eligible"] is True


def test_vision_model_node_selects_only_probe_eligible_model(fake_client):
    fake_client.responses = {
        "/v1/models": {"data": [{"id": "qwen-vl"}, {"id": "glm-v"}]},
        "/v1/capabilities": {
            "models": [
                {
                    "id": "qwen-vl",
                    "tasks": ["image-understanding"],
                    "available_tasks": ["image-understanding"],
                },
                {
                    "id": "glm-v",
                    "tasks": ["image-understanding"],
                    "available_tasks": [],
                },
            ]
        },
    }
    selected, available, metadata = WerkVisionModelsNode().select(
        WerkConnection("http://werk"), 1, "", True
    )
    assert selected == "qwen-vl"
    assert available == "qwen-vl"
    assert json.loads(metadata)["declared"] == ["qwen-vl", "glm-v"]


def test_vision_config_contains_only_supported_chat_controls():
    config = build_vision_config(
        temperature=0.1,
        top_p=0.8,
        max_completion_tokens=2048,
        seed=1112,
        image_detail="high",
        stop_sequences_json='["<END>"]',
    )
    assert isinstance(config, WerkVisionConfig)
    assert vision_config_payload(config) == {
        "request_fields": {
            "temperature": 0.1,
            "top_p": 0.8,
            "max_completion_tokens": 2048,
            "seed": 1112,
            "stop": ["<END>"],
        },
        "image_detail": "high",
    }
    assert "routing" not in vision_config_payload(config)


def test_vision_input_byte_limit_environment_is_validated(monkeypatch):
    monkeypatch.setenv("WERK_MAX_VISION_INPUT_BYTES", "1234")
    assert environment_max_vision_input_bytes() == 1234
    monkeypatch.setenv("WERK_MAX_VISION_INPUT_BYTES", "0")
    with pytest.raises(ValueError, match="greater than zero"):
        environment_max_vision_input_bytes()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"temperature": float("nan")}, "temperature"),
        ({"top_p": 1.1}, "top_p"),
        ({"max_completion_tokens": 0}, "at least 1"),
        ({"image_detail": "original"}, "auto, low, or high"),
        ({"stop_sequences_json": '{"bad":true}'}, "array of strings"),
    ],
)
def test_vision_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_vision_config(**kwargs)


def test_vision_request_preserves_image_batch_order_in_one_user_message():
    images = torch.stack(
        [torch.zeros((1, 1, 3)), torch.ones((1, 1, 3))]
    )
    request = build_vision_request(
        model="qwen-vl",
        prompt="Inspect both screenshots in order.",
        images=images,
        system_prompt="Act as a visual QA reviewer.",
        config=build_vision_config(image_detail="high", seed=1112),
    )
    assert request["stream"] is False
    assert request["model"] == "qwen-vl"
    assert request["seed"] == 1112
    assert request["messages"][0] == {
        "role": "system",
        "content": "Act as a visual QA reviewer.",
    }
    user = request["messages"][1]
    assert user["role"] == "user"
    assert user["content"][-1] == {
        "type": "text",
        "text": "Inspect both screenshots in order.",
    }
    image_parts = user["content"][:-1]
    assert [part["type"] for part in image_parts] == ["image_url", "image_url"]
    assert all(part["image_url"]["detail"] == "high" for part in image_parts)
    assert all(
        part["image_url"]["url"].startswith("data:image/png;base64,")
        for part in image_parts
    )
    assert image_parts[0]["image_url"]["url"] != image_parts[1]["image_url"]["url"]
    assert "routing" not in request
    assert "parameters" not in request


def test_vision_request_requires_active_typed_config():
    with pytest.raises(TypeError, match="WerkVisionConfig"):
        build_vision_request(
            model="qwen-vl",
            prompt="Inspect.",
            images=torch.zeros((1, 1, 1, 3)),
            config=None,
        )


def test_vision_execution_posts_chat_contract_and_sanitizes_metadata(fake_client):
    fake_client.responses["/v1/chat/completions"] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1112,
        "model": "qwen-vl",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The button is missing."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    request = {"model": "qwen-vl", "messages": [], "stream": False}
    result = execute_vision_request(WerkConnection("http://werk"), request)
    assert fake_client.posted == ("/v1/chat/completions", request)
    assert result[0] == "The button is missing."
    metadata = json.loads(result[1])
    assert metadata["choice"]["finish_reason"] == "stop"
    assert "choices" not in metadata
    assert result[2:] == ("chatcmpl-1", "stop")


def test_vision_node_surfaces_are_typed_and_non_streaming():
    config_inputs = WerkVisionConfigNode.INPUT_TYPES()["required"]
    analyze_inputs = WerkVisionAnalyzeNode.INPUT_TYPES()
    assert set(config_inputs) == {
        "temperature",
        "top_p",
        "max_completion_tokens",
        "seed",
        "image_detail",
        "stop_sequences_json",
    }
    assert analyze_inputs["required"]["images"] == ("IMAGE",)
    assert analyze_inputs["required"]["config"] == ("WERK_VISION_CONFIG",)
    assert WerkVisionAnalyzeNode.RETURN_NAMES[0] == "analysis"
