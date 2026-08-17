import pytest

from ..client import WerkApiError
from ..routes import discover_connection


class FakeClient:
    def __init__(self, connection):
        self.connection = connection

    def get_json(self, path, query=None):
        del query
        if path == "/v1/models":
            return {
                "object": "list",
                "data": [
                    {"id": "image-ready", "object": "model"},
                    {"id": "image-unavailable", "object": "model"},
                    {"id": "video-ready", "object": "model"},
                    {"id": "video-i2v", "object": "model"},
                    {"id": "text", "object": "model"},
                ],
            }
        if path == "/v1/capabilities":
            return {
                "object": "werk.capabilities",
                "models": [
                    {
                        "id": "image-ready",
                        "tasks": ["image_generation"],
                        "available_tasks": ["image_generation"],
                    },
                    {
                        "id": "image-unavailable",
                        "tasks": ["image_generation"],
                        "available_tasks": [],
                    },
                    {
                        "id": "text",
                        "tasks": ["text_generation"],
                        "available_tasks": ["text_generation"],
                    },
                    {
                        "id": "video-ready",
                        "tasks": ["video_generation"],
                        "available_tasks": ["video_generation"],
                    },
                    {
                        "id": "video-i2v",
                        "tasks": ["image_to_video"],
                        "available_tasks": ["image_to_video"],
                    },
                ],
            }
        raise AssertionError(path)


def test_connection_discovery_returns_safe_status_and_model_lists():
    result = discover_connection(
        {
            "server_url": "http://127.0.0.1:11434/",
            "api_key": "never-return-this",
            "timeout_seconds": 30,
            "verify_tls": True,
        },
        client_factory=FakeClient,
    )
    assert result["ok"] is True
    assert result["server_url"] == "http://127.0.0.1:11434"
    assert result["authentication_configured"] is True
    assert result["models"] == [
        "image-ready",
        "image-unavailable",
        "video-ready",
        "video-i2v",
        "text",
    ]
    assert result["image_models"] == {
        "declared": ["image-ready", "image-unavailable"],
        "available": ["image-ready"],
    }
    assert result["video_models"] == {
        "declared": ["video-ready", "video-i2v"],
        "available": ["video-ready", "video-i2v"],
        "by_task": {
            "video-generation": {
                "declared": ["video-ready"],
                "available": ["video-ready"],
            },
            "image-to-video": {
                "declared": ["video-i2v"],
                "available": ["video-i2v"],
            },
        },
    }
    assert result["status"] == "Connected · 5 models · 1 image · 2 videos"
    assert "never-return-this" not in repr(result)


def test_connection_discovery_tolerates_optional_capabilities_failure():
    class PartialClient(FakeClient):
        def get_json(self, path, query=None):
            if path == "/v1/capabilities":
                raise WerkApiError("GET", "http://werk/v1/capabilities", 404, "not supported")
            return super().get_json(path, query)

    result = discover_connection(
        {"server_url": "http://werk", "api_key": "", "timeout_seconds": 30, "verify_tls": True},
        client_factory=PartialClient,
    )
    assert result["ok"] is True
    assert result["image_models"] == {"declared": [], "available": []}
    assert result["video_models"]["declared"] == []
    assert "not supported" in result["warning"]


@pytest.mark.parametrize(
    "payload,message",
    [
        ({}, "must not be empty"),
        ({"server_url": "ftp://werk"}, "http or https"),
        ({"server_url": "http://werk", "api_key": 123}, "api_key must be a string"),
        ({"server_url": "http://werk", "timeout_seconds": True}, "timeout_seconds must be an integer"),
        ({"server_url": "http://werk", "verify_tls": "yes"}, "verify_tls must be a boolean"),
    ],
)
def test_connection_discovery_validates_frontend_payload(payload, message):
    with pytest.raises(ValueError, match=message):
        discover_connection(payload, client_factory=FakeClient)
