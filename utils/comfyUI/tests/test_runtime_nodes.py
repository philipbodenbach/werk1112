import json
from typing import ClassVar

import pytest

from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, runtime_nodes
from ..config import WerkConnection
from ..runtime_nodes import (
    WerkDecodeNode,
    WerkExpertControlNode,
    WerkMemoryStatusNode,
    WerkPersistencePolicy,
    WerkPersistencePolicyNode,
    WerkPrefillNode,
    WerkRuntimeExpertsNode,
    WerkRuntimeInfoNode,
    WerkRuntimeStatesNode,
    WerkStateControlNode,
    WerkStateHandoff,
    WerkStatePruneNode,
)

ALL_STATUSES = (
    "supported",
    "unsupported",
    "unavailable",
    "experimental",
    "externally_managed",
    "metadata_only",
)


def capability(capability_id, status="supported"):
    return {
        "id": capability_id,
        "status": status,
        "detail": status,
        "operations": ["read"],
    }


def state(state_id="state_1", tier="ram", pinned=False):
    return {
        "id": state_id,
        "model_id": "model",
        "tier": tier,
        "status": "ready",
        "bytes": 100,
        "created_unix_ms": 1,
        "last_accessed_unix_ms": 2,
        "expires_unix_ms": None,
        "pinned": pinned,
        "backend": "test",
        "reusable": True,
    }


def expert(expert_id="expert_1", tier="vram", pinned=False):
    return {
        "id": expert_id,
        "model_id": "model",
        "tier": tier,
        "bytes": 256,
        "hotness": 1.5,
        "pinned": pinned,
        "last_used_unix_ms": 9,
    }


class FakeProtocolClient:
    instances: ClassVar[list] = []
    info_response: ClassVar[dict] = {
        "service": "werk1112",
        "service_version": "1.5.1",
        "protocol": {"major": 1, "minor": 0},
        "active_backend": "test",
        "limits": {
            "max_page_size": 100,
            "max_state_ids_per_operation": 100,
            "max_expert_ids_per_operation": 256,
            "max_request_bytes": 1048576,
            "max_handoff_bytes": 4096,
            "max_ttl_seconds": 2592000,
        },
    }
    capabilities_response: ClassVar[dict] = {
        "capabilities": [
            capability("runtime.experts.residency"),
            capability("runtime.pd.prefill"),
            capability("runtime.pd.decode"),
            capability("runtime.pd.handoff"),
        ]
    }
    states_response: ClassVar[dict] = {
        "states": [state()],
        "next_cursor": "cursor_2",
    }
    action_response: ClassVar[dict] = {
        "state": state(pinned=True),
        "changed": True,
        "dry_run": False,
    }
    prune_response: ClassVar[dict] = {
        "matched": 2,
        "removed": 0,
        "bytes": 200,
        "dry_run": True,
    }
    memory_response: ClassVar[dict] = {
        "observed_at_unix_ms": 10,
        "overall_pressure": "soft",
        "topology": "discrete",
        "host": {
            "capacity_bytes": 1000,
            "available_bytes": 600,
            "managed_bytes": 300,
            "reserved_bytes": 100,
            "pressure": "soft",
        },
        "accelerator": {
            "capacity_bytes": None,
            "available_bytes": None,
            "managed_bytes": 0,
            "reserved_bytes": 0,
            "pressure": "unknown",
        },
        "last_action_unix_ms": None,
        "counters": {"demotions": 1},
    }
    experts_response: ClassVar[dict] = {
        "experts": [expert()],
        "next_cursor": "expert_cursor_2",
    }
    expert_action_response: ClassVar[dict] = {
        "experts": [expert(pinned=True)],
        "changed": 1,
        "dry_run": True,
    }
    prefill_response: ClassVar[dict] = {
        "handoff": "h" * 40,
        "state_id": "state_1",
        "prompt_tokens": 12,
        "reused": False,
        "tier": "ram",
        "expires_unix_ms": 500,
    }
    decode_response: ClassVar[dict] = {
        "text": "hello",
        "handoff": "u" * 40,
        "completion_tokens": 3,
        "finish_reason": "stop",
    }

    def __init__(self, connection):
        self.connection = connection
        self.calls = []
        self.__class__.instances.append(self)

    def info(self):
        self.calls.append(("info",))
        return self.info_response

    def capabilities(self):
        self.calls.append(("capabilities",))
        return self.capabilities_response

    def states(self, query):
        self.calls.append(("states", query))
        return self.states_response

    def state_action(self, state_id, payload):
        self.calls.append(("state_action", state_id, payload))
        return self.action_response

    def prune_states(self, payload):
        self.calls.append(("prune_states", payload))
        return self.prune_response

    def memory(self):
        self.calls.append(("memory",))
        return self.memory_response

    def experts(self, query):
        self.calls.append(("experts", query))
        return self.experts_response

    def expert_action(self, payload):
        self.calls.append(("expert_action", payload))
        return self.expert_action_response

    def prefill(self, payload):
        self.calls.append(("prefill", payload))
        return self.prefill_response

    def decode(self, payload, *, handoff_secret):
        self.calls.append(("decode", payload, handoff_secret))
        return self.decode_response


@pytest.fixture
def fake_protocol(monkeypatch):
    FakeProtocolClient.instances = []
    FakeProtocolClient.capabilities_response = {
        "capabilities": [
            capability("runtime.experts.residency"),
            capability("runtime.pd.prefill"),
            capability("runtime.pd.decode"),
            capability("runtime.pd.handoff"),
        ]
    }
    monkeypatch.setattr(runtime_nodes, "WerkProtocolClient", FakeProtocolClient)
    return FakeProtocolClient


@pytest.fixture
def connection():
    return WerkConnection("http://werk.invalid", "secret")


def test_runtime_nodes_are_additive_and_have_custom_handoff_socket():
    expected = {
        "WerkRuntimeInfo",
        "WerkPersistencePolicy",
        "WerkRuntimeStates",
        "WerkStateControl",
        "WerkStatePrune",
        "WerkMemoryStatus",
        "WerkRuntimeExperts",
        "WerkExpertControl",
        "WerkPrefill",
        "WerkDecode",
    }
    assert expected <= set(NODE_CLASS_MAPPINGS)
    assert NODE_CLASS_MAPPINGS["WerkRuntimeInfo"] is WerkRuntimeInfoNode
    assert NODE_CLASS_MAPPINGS["WerkDecode"] is WerkDecodeNode
    assert NODE_CLASS_MAPPINGS["WerkRuntimeExperts"] is WerkRuntimeExpertsNode
    assert NODE_CLASS_MAPPINGS["WerkExpertControl"] is WerkExpertControlNode
    assert expected <= set(NODE_DISPLAY_NAME_MAPPINGS)
    assert WerkPrefillNode.RETURN_TYPES[0] == "WERK_STATE_HANDOFF"
    assert WerkDecodeNode.INPUT_TYPES()["required"]["handoff"] == (
        "WERK_STATE_HANDOFF",
    )


def test_runtime_info_preserves_all_six_capability_statuses(
    fake_protocol, connection
):
    fake_protocol.capabilities_response = {
        "capabilities": [capability(f"runtime.{value}", value) for value in ALL_STATUSES]
    }
    runtime, summary, info_json, capabilities_json = WerkRuntimeInfoNode().discover(
        connection, 0
    )
    assert runtime.info["active_backend"] == "test"
    assert {item["status"] for item in runtime.capabilities} == set(ALL_STATUSES)
    assert "protocol 1.0" in summary
    assert json.loads(info_json)["service"] == "werk1112"
    assert len(json.loads(capabilities_json)["capabilities"]) == 6
    assert "secret" not in repr(runtime)


def test_persistence_policy_defaults_and_ttl_omission():
    inputs = WerkPersistencePolicyNode.INPUT_TYPES()["required"]
    assert inputs["mode"][1]["default"] == "auto"
    assert inputs["reuse"][1]["default"] == "prefer"
    assert inputs["ttl_seconds"][1]["default"] == 0
    assert inputs["pin"][1]["default"] is False

    policy, rendered = WerkPersistencePolicyNode().configure(
        "auto", "prefer", 0, False
    )
    assert policy.payload() == {"mode": "auto", "reuse": "prefer", "pin": False}
    assert "ttl_seconds" not in json.loads(rendered)
    persistent, _ = WerkPersistencePolicyNode().configure(
        "disk", "required", 60, True
    )
    assert persistent.payload()["ttl_seconds"] == 60
    with pytest.raises(ValueError, match="unknown persistence mode"):
        WerkPersistencePolicy("forever")


def test_runtime_states_uses_only_versioned_discovery_and_filter(
    fake_protocol, connection
):
    states_json, ids, cursor, count = WerkRuntimeStatesNode().list_states(
        connection, "model", "ram", 25, "cursor_1", 9
    )
    client = fake_protocol.instances[-1]
    assert client.calls == [
        ("capabilities",),
        ("info",),
        (
            "states",
            {
                "model_id": "model",
                "tier": "ram",
                "limit": 25,
                "cursor": "cursor_1",
            },
        ),
    ]
    assert json.loads(states_json)[0]["id"] == "state_1"
    assert ids == "state_1"
    assert cursor == "cursor_2"
    assert count == 1


def test_runtime_states_enforces_the_advertised_page_bound(
    fake_protocol, connection, monkeypatch
):
    limited_info = {
        **fake_protocol.info_response,
        "limits": {
            **fake_protocol.info_response["limits"],
            "max_page_size": 10,
        },
    }
    monkeypatch.setattr(fake_protocol, "info_response", limited_info)
    with pytest.raises(ValueError, match="state page limit of at most 10"):
        WerkRuntimeStatesNode().list_states(
            connection, "", "all", 11, "", 0
        )
    assert fake_protocol.instances[-1].calls == [
        ("capabilities",),
        ("info",),
    ]


def test_state_control_quotes_id_and_sends_explicit_safety_fields(
    fake_protocol, connection
):
    result, changed, dry_run, summary = WerkStateControlNode().control(
        connection, "state/one", "promote", "vram", False, True
    )
    client = fake_protocol.instances[-1]
    assert client.calls[-1] == (
        "state_action",
        "state%2Fone",
        {
            "action": "promote",
            "dry_run": False,
            "allow_experimental": True,
            "target_tier": "vram",
        },
    )
    assert json.loads(result)["pinned"] is True
    assert changed is True and dry_run is False
    assert "changed" in summary
    with pytest.raises(ValueError, match="ram or disk"):
        WerkStateControlNode().control(
            connection, "state", "demote", "unchanged", True, False
        )


@pytest.mark.parametrize(
    ("action", "target", "message"),
    [
        ("promote", "disk", "vram or ram"),
        ("promote", "unchanged", "vram or ram"),
        ("demote", "vram", "ram or disk"),
        ("demote", "unchanged", "ram or disk"),
        ("pin", "ram", "only valid for promote and demote"),
        ("unpin", "disk", "only valid for promote and demote"),
        ("evict", "vram", "only valid for promote and demote"),
    ],
)
def test_state_control_rejects_invalid_action_tier_combinations(
    fake_protocol, connection, action, target, message
):
    with pytest.raises(ValueError, match=message):
        WerkStateControlNode().control(
            connection, "state", action, target, True, False
        )
    assert fake_protocol.instances == []


def test_state_control_requires_boolean_safety_switches(connection):
    with pytest.raises(TypeError, match="must be booleans"):
        WerkStateControlNode().control(
            connection, "state", "pin", "unchanged", "false", False
        )


def test_state_prune_defaults_to_dry_run_and_requires_bounded_selectors(
    fake_protocol, connection
):
    inputs = WerkStatePruneNode.INPUT_TYPES()["required"]
    assert inputs["dry_run"][1]["default"] is True
    assert inputs["confirm_all"][1]["default"] is False

    matched, removed, bytes_value, dry_run, summary, result = (
        WerkStatePruneNode().prune(
            connection,
            "ids",
            "state_1\nstate_2",
            "",
            "all",
            0,
            False,
            True,
        )
    )
    client = fake_protocol.instances[-1]
    assert client.calls[-1] == (
        "prune_states",
        {
            "selector": {"kind": "ids", "ids": ["state_1", "state_2"]},
            "dry_run": True,
        },
    )
    assert (matched, removed, bytes_value, dry_run) == (2, 0, "200", True)
    assert "would remove" in summary
    assert "would remove 2" in summary
    assert json.loads(result)["dry_run"] is True

    with pytest.raises(ValueError, match="at least one constraint"):
        WerkStatePruneNode().prune(
            connection, "filter", "", "", "all", 0, False, True
        )
    with pytest.raises(ValueError, match="requires confirm_all"):
        WerkStatePruneNode().prune(
            connection, "all", "", "", "all", 0, False, True
        )


def test_memory_status_keeps_unknown_values_explicit(fake_protocol, connection):
    pressure, host, accelerator, status_json = WerkMemoryStatusNode().status(
        connection, 0
    )
    assert pressure == "soft"
    assert "capacity=1000" in host
    assert "capacity=unknown" in accelerator
    assert json.loads(status_json)["accelerator"]["pressure"] == "unknown"


def test_runtime_experts_is_bounded_capability_gated_telemetry(
    fake_protocol, connection
):
    experts_json, ids, cursor, count = WerkRuntimeExpertsNode().list_experts(
        connection, "model", "vram", 25, "expert_cursor_1", False, 7
    )
    client = fake_protocol.instances[-1]
    assert client.calls == [
        ("capabilities",),
        ("info",),
        (
            "experts",
            {
                "model_id": "model",
                "tier": "vram",
                "limit": 25,
                "cursor": "expert_cursor_1",
                "allow_experimental": False,
            },
        ),
    ]
    assert json.loads(experts_json) == [expert()]
    assert (ids, cursor, count) == ("expert_1", "expert_cursor_2", 1)

    fake_protocol.capabilities_response = {
        "capabilities": [
            capability("runtime.experts.residency", "externally_managed")
        ]
    }
    assert WerkRuntimeExpertsNode().list_experts(
        connection, "", "all", 1, "", False, 0
    )[-1] == 1


@pytest.mark.parametrize("status", ["unsupported", "unavailable", "metadata_only"])
def test_runtime_experts_fail_closed_for_non_readable_statuses(
    fake_protocol, connection, status
):
    fake_protocol.capabilities_response = {
        "capabilities": [capability("runtime.experts.residency", status)]
    }
    with pytest.raises(ValueError, match=f"is {status}"):
        WerkRuntimeExpertsNode().list_experts(
            connection, "", "all", 1, "", False, 0
        )
    assert fake_protocol.instances[-1].calls == [("capabilities",)]


def test_runtime_experts_require_explicit_opt_in_for_experimental_status(
    fake_protocol, connection
):
    fake_protocol.capabilities_response = {
        "capabilities": [
            capability("runtime.experts.residency", "experimental")
        ]
    }
    with pytest.raises(ValueError, match="experimental"):
        WerkRuntimeExpertsNode().list_experts(
            connection, "", "all", 1, "", False, 0
        )
    assert WerkRuntimeExpertsNode().list_experts(
        connection, "", "all", 1, "", True, 0
    )[-1] == 1


def test_runtime_experts_enforce_the_advertised_page_bound(
    fake_protocol, connection
):
    with pytest.raises(ValueError, match="page limit of at most 100"):
        WerkRuntimeExpertsNode().list_experts(
            connection, "", "all", 101, "", False, 0
        )
    assert fake_protocol.instances[-1].calls == [
        ("capabilities",),
        ("info",),
    ]


def test_expert_control_uses_explicit_ids_and_dry_run_by_default(
    fake_protocol, connection
):
    inputs = WerkExpertControlNode.INPUT_TYPES()["required"]
    assert inputs["dry_run"][1]["default"] is True
    assert inputs["allow_experimental"][1]["default"] is False

    experts_json, changed, dry_run, summary = WerkExpertControlNode().control(
        connection,
        "model",
        '["expert_1", "expert.2"]',
        "prefetch",
        "vram",
        True,
        True,
    )
    client = fake_protocol.instances[-1]
    assert client.calls == [
        ("capabilities",),
        ("info",),
        (
            "expert_action",
            {
                "model_id": "model",
                "expert_ids": ["expert_1", "expert.2"],
                "action": "prefetch",
                "target_tier": "vram",
                "dry_run": True,
                "allow_experimental": True,
            },
        ),
    ]
    assert json.loads(experts_json)[0]["pinned"] is True
    assert (changed, dry_run) == (1, True)
    assert "would change 1" in summary


def test_expert_control_validates_action_tier_ids_and_server_bound(
    fake_protocol, connection, monkeypatch
):
    node = WerkExpertControlNode()
    with pytest.raises(ValueError, match="vram or ram"):
        node.control(connection, "model", "expert_1", "prefetch", "external", True, False)
    with pytest.raises(ValueError, match="only valid for prefetch"):
        node.control(connection, "model", "expert_1", "pin", "ram", True, False)
    with pytest.raises(ValueError, match="duplicate"):
        node.control(
            connection,
            "model",
            "expert_1\nexpert_1",
            "pin",
            "unchanged",
            True,
            False,
        )
    with pytest.raises(ValueError, match="invalid opaque"):
        node.control(connection, "model", "../one", "evict", "unchanged", True, False)
    with pytest.raises(ValueError, match="safe characters"):
        node.control(
            connection,
            "../model",
            "expert_1",
            "pin",
            "unchanged",
            True,
            False,
        )

    limited_info = {
        **fake_protocol.info_response,
        "limits": {
            **fake_protocol.info_response["limits"],
            "max_expert_ids_per_operation": 1,
        },
    }
    monkeypatch.setattr(fake_protocol, "info_response", limited_info)
    with pytest.raises(ValueError, match="at most 1 expert IDs"):
        node.control(
            connection,
            "model",
            "expert_1\nexpert_2",
            "pin",
            "unchanged",
            True,
            False,
        )
    assert all(
        call[0] != "expert_action" for call in fake_protocol.instances[-1].calls
    )


def test_expert_control_rejects_externally_managed_and_honors_experimental_opt_in(
    fake_protocol, connection
):
    fake_protocol.capabilities_response = {
        "capabilities": [
            capability("runtime.experts.residency", "externally_managed")
        ]
    }
    with pytest.raises(ValueError, match="externally_managed"):
        WerkExpertControlNode().control(
            connection, "model", "expert_1", "pin", "unchanged", True, False
        )
    assert fake_protocol.instances[-1].calls == [("capabilities",)]

    fake_protocol.capabilities_response = {
        "capabilities": [
            capability("runtime.experts.residency", "experimental")
        ]
    }
    WerkExpertControlNode().control(
        connection, "model", "expert_1", "pin", "unchanged", True, True
    )
    assert fake_protocol.instances[-1].calls[-1][0] == "expert_action"


def test_prefill_builds_typed_policy_and_never_serializes_handoff(
    fake_protocol, connection
):
    policy = WerkPersistencePolicy("disk", "required", 60, True)
    result = WerkPrefillNode().prefill(
        connection,
        "model",
        "messages",
        "ignored",
        '[{"role":"user","content":"hello"}]',
        True,
        policy,
    )
    handoff, state_id, tokens, reused, tier, expires, metadata = result
    assert isinstance(handoff, WerkStateHandoff)
    assert "h" * 40 not in repr(handoff)
    with pytest.raises(TypeError):
        json.dumps(handoff)
    assert "h" * 40 not in metadata
    assert (state_id, tokens, reused, tier, expires) == (
        "state_1",
        12,
        False,
        "ram",
        500,
    )
    call = fake_protocol.instances[-1].calls[-1]
    assert call == (
        "prefill",
        {
            "model_id": "model",
            "input": {
                "type": "messages",
                "messages": [{"role": "user", "content": "hello"}],
            },
            "policy": {
                "mode": "disk",
                "reuse": "required",
                "pin": True,
                "ttl_seconds": 60,
            },
            "allow_experimental": True,
        },
    )


def test_prefill_omits_an_unconnected_policy_for_server_defaults(
    fake_protocol, connection
):
    WerkPrefillNode().prefill(
        connection,
        "model",
        "text",
        "hello",
        "[]",
        True,
    )
    assert fake_protocol.instances[-1].calls[-1] == (
        "prefill",
        {
            "model_id": "model",
            "input": {"type": "text", "text": "hello"},
            "allow_experimental": True,
        },
    )


def test_prefill_and_decode_fail_closed_on_capabilities(
    fake_protocol, connection
):
    fake_protocol.capabilities_response = {
        "capabilities": [capability("runtime.pd.prefill", "unsupported")]
    }
    with pytest.raises(ValueError, match="is unsupported"):
        WerkPrefillNode().prefill(
            connection, "model", "text", "hello", "[]", False
        )
    assert all(
        call[0] != "prefill" for call in fake_protocol.instances[-1].calls
    )

    fake_protocol.capabilities_response = {
        "capabilities": [
            capability("runtime.pd.decode", "experimental"),
            capability("runtime.pd.handoff"),
        ]
    }
    handoff = WerkStateHandoff("x" * 40)
    with pytest.raises(ValueError, match="experimental"):
        WerkDecodeNode().decode(
            connection, handoff, 1, -1, -1, -1, "[]", False
        )


def test_prefill_can_trigger_server_side_experimental_capability_probe(
    fake_protocol, connection
):
    fake_protocol.capabilities_response = {
        "capabilities": [
            capability("runtime.pd.prefill", "unavailable"),
            capability("runtime.pd.handoff", "unavailable"),
        ]
    }
    handoff, *_ = WerkPrefillNode().prefill(
        connection, "model", "text", "hello", "[]", True
    )
    assert isinstance(handoff, WerkStateHandoff)
    assert fake_protocol.instances[-1].calls[-1][0] == "prefill"

    with pytest.raises(ValueError, match="is unavailable"):
        WerkPrefillNode().prefill(
            connection, "model", "text", "hello", "[]", False
        )


def test_decode_passes_opaque_handoff_and_returns_only_safe_metadata(
    fake_protocol, connection
):
    token = "x" * 40
    handoff = WerkStateHandoff(token)
    text, updated, reason, tokens, metadata = WerkDecodeNode().decode(
        connection, handoff, 64, -1, -1, -1, '["END"]', False
    )
    call = fake_protocol.instances[-1].calls[-1]
    assert call == (
        "decode",
        {
            "handoff": token,
            "max_tokens": 64,
            "stop": ["END"],
            "allow_experimental": False,
        },
        token,
    )
    assert text == "hello" and reason == "stop" and tokens == 3
    assert isinstance(updated, WerkStateHandoff)
    assert "u" * 40 not in repr(updated)
    assert token not in metadata and "u" * 40 not in metadata
    assert json.loads(metadata)["has_updated_handoff"] is True
    with pytest.raises(TypeError, match="WERK_STATE_HANDOFF"):
        WerkDecodeNode().decode(
            connection, token, 1, -1, -1, -1, "[]", False
        )
