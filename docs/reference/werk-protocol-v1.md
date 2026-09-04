---
title: Werk Protocol 1.0 HTTP reference
description: Versioned HTTP/JSON contract for Werk1112 runtime residency, persistence, memory, expert discovery and prefill/decode.
---

# Werk Protocol 1.0 HTTP reference

[Back to the documentation index](../documentation.md)

Werk Protocol 1.0 is the versioned runtime-control API under `/werk/v1`. It is
separate from the OpenAI-compatible and Werk-native inference routes under
`/v1`; adding it does not change those established contracts.

The architecture, backend matrix and persistence guarantees are documented in
[Runtime persistence and memory architecture](../concepts/runtime-persistence-and-memory.md).

## Connection and authentication

The default server is:

~~~text
http://127.0.0.1:11434
~~~

When `werk serve` has API keys configured, every protocol route accepts either:

~~~http
Authorization: Bearer sk-werk-example
~~~

or:

~~~http
X-API-Key: sk-werk-example
~~~

The authenticated key is converted to an opaque principal for runtime-state
isolation. It is not sent to the transport-neutral service or used as a
filesystem name. When the server is deliberately started without configured
keys, all requests share the `local` principal.

Werk does not terminate TLS. The bundled client accepts only an explicit
`http://HOST:PORT` URL, so it is intended for loopback or a trusted private
hop. Do not send a key over an untrusted network.

## Version and envelopes

Clients should advertise the JSON representation and the greatest Werk
Protocol version they can consume:

~~~http
Accept: application/json
X-Werk-Protocol-Version: 1.0
~~~

For compatibility with initial 1.0 clients, either request header may be
omitted. If `Accept` is present, it must allow `application/json`,
`application/*` or `*/*` with a non-zero quality value. Every supplied
`X-Werk-Protocol-Version` value must be able to consume the server's 1.0
response. Otherwise the server returns HTTP 406 with the typed
`incompatible_protocol` error.

Every success response has this envelope:

~~~json
{
  "protocol": {"major": 1, "minor": 0},
  "request_id": "req_opaque-value",
  "data": {}
}
~~~

Every protocol error has this envelope:

~~~json
{
  "protocol": {"major": 1, "minor": 0},
  "request_id": "req_opaque-value",
  "error": {
    "code": "invalid_request",
    "message": "human-readable explanation",
    "retryable": false
  }
}
~~~

`X-Werk-Request-Id` repeats the response request ID. Every protocol response,
including an error, declares `X-Werk-Protocol-Version: 1.0`,
`Content-Type: application/json`, `Cache-Control: no-store`, `Vary: accept`
and `X-Content-Type-Options: nosniff`.

Compatibility requires the same major version. The bundled 1.x client accepts
a producer minor version no newer than the client's minor version and rejects
an incompatible envelope before returning its payload. It also verifies a
response version header when present and rejects disagreement with the JSON
envelope. A missing response version header remains accepted for compatibility
with an earlier 1.0 server; the envelope itself is always required.

## Discovery

Discovery is the required first step. A client must use the reported status,
not infer support from `active_backend`.

### GET /werk/v1/info

Returns the service, active adapter and negotiated limits:

~~~json
{
  "service": "werk1112",
  "service_version": "1.4.0",
  "protocol": {"major": 1, "minor": 0},
  "active_backend": "llama-server-cpu",
  "limits": {
    "max_page_size": 100,
    "max_state_ids_per_operation": 100,
    "max_expert_ids_per_operation": 256,
    "max_request_bytes": 1048576,
    "max_handoff_bytes": 4096,
    "max_ttl_seconds": 2592000
  }
}
~~~

The body is the `data` member of the common success envelope. Examples on this
page show `data` values alone where that keeps them readable.

### GET /werk/v1/capabilities

Returns an array of:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable capability identifier such as `runtime.pd.prefill`. |
| `status` | enum | `supported`, `unsupported`, `unavailable`, `experimental`, `externally_managed`, or `metadata_only`. |
| `detail` | string | Current backend-specific reason or scope. |
| `operations` | string array | Operation names scoped by this entry. The status remains authoritative; a listed operation can still be presently `unavailable`. |

The [capability semantics and production matrix](../concepts/runtime-persistence-and-memory.md#capability-status-semantics)
are normative for client behavior. `experimental` requires an effective
request-level opt-in. A client can send it explicitly; for Prefill only, a
server started with `--persistence` can supply it when the member is absent.
That server option does not opt other protocol operations in.

`runtime.model_residency` is deliberately independent of every
`runtime.state.*` capability. Its `automatic_reuse` operation reports that an
exact model or pipeline can remain loaded between ordinary requests. It does
not produce a `state_id` or handoff and does not promise prompt/KV reuse,
durability, or cross-restart recovery. A local Werk-owned process or worker can
report `supported`; a remote service can report `externally_managed`; a
one-shot runner can report `unsupported`; and an adapter with a temporarily
missing runtime or process prerequisite can report `unavailable`. The current
backend-specific `detail` remains authoritative.

For vLLM, `runtime.state.prefix_cache` is `externally_managed` with the sole
operation `automatic_reuse` only when at least one active Werk-started process
is local, no active process is remote, and every active process explicitly
enables APC without an effective disable flag. No process is `unavailable`;
all-disabled is `unsupported`; remote, mixed or ambiguous evidence is
`metadata_only`. None of those statuses makes the named-state, prune,
persistence, handoff or decode-from-state endpoints operational.

An active local vLLM process separately reports model residency as
`supported`; an active remote endpoint reports it as `externally_managed`.
`werk serve --persistence` may supply native APC arguments only when Werk
starts vLLM locally and the operator has not explicitly selected an APC flag.
It sends no launch setting to remote vLLM.

## Route inventory

| Method | Path | Request | `data` response |
| --- | --- | --- | --- |
| GET | `/werk/v1/info` | none | Runtime info and limits |
| GET | `/werk/v1/capabilities` | none | Capability array |
| GET | `/werk/v1/memory` | none | Memory telemetry and accounting |
| GET | `/werk/v1/states` | query filters | One page of runtime states |
| POST | `/werk/v1/states/{id}/actions` | State action | Updated/projected state |
| POST | `/werk/v1/states/prune` | Explicit selector | Match/removal summary |
| GET | `/werk/v1/experts` | query filters | One page of expert metadata |
| POST | `/werk/v1/experts/actions` | Explicit expert IDs/action | Expert action summary |
| POST | `/werk/v1/prefill` | Input and persistence policy | Opaque handoff and state metadata |
| POST | `/werk/v1/decode` | Opaque handoff and decode options | Completion and optional next handoff |

All five POST routes have a fixed 1 MiB HTTP body limit. Individual DTOs have
tighter semantic limits where described below. Unknown fields are rejected on
the mutation, prefill and decode request DTOs.

## Memory

### GET /werk/v1/memory

Example `data`:

~~~json
{
  "observed_at_unix_ms": 1788444000000,
  "overall_pressure": "normal",
  "topology": "discrete",
  "host": {
    "capacity_bytes": 68719476736,
    "available_bytes": 42949672960,
    "managed_bytes": 0,
    "reserved_bytes": 0,
    "pressure": "normal"
  },
  "accelerator": {
    "capacity_bytes": null,
    "available_bytes": null,
    "managed_bytes": 0,
    "reserved_bytes": 0,
    "pressure": "unknown"
  },
  "last_action_unix_ms": null,
  "counters": {
    "active_reservations": 0,
    "managed_allocations": 0,
    "pressure_actions_in_flight": 0,
    "completed_demotions": 0,
    "completed_evictions": 0,
    "failed_releases": 0,
    "orphaned_release_bytes": 0,
    "telemetry_errors": 0,
    "backend_cleanup_failures": 0,
    "backend_cleanup_latched": 0
  }
}
~~~

`capacity_bytes` and `available_bytes` are nullable because Werk refuses to
invent missing telemetry. Topology is currently `discrete`, `unified` or
`unknown`; pressure is `normal`, `soft`, `hard`, `emergency` or `unknown`.
Only allocations made through an adapter with bounded pre-load requirements
appear as managed/reserved bytes.

`failed_releases` counts backend cleanup failures for managed allocations.
Werk then retains the allocation in accounting and pins it against policy
movement instead of claiming that capacity was freed. Reservation admission
uses projected utilization: after a rejection, one bounded relief plan may
demote or evict eligible least-recently-used state until the requested bytes
would remain below the hard threshold, even if the current pre-request
pressure is still `normal` or `soft`. `orphaned_release_bytes` is the known
logical size whose backend cleanup could not be confirmed; unknown sizes add
zero rather than an estimate. `telemetry_errors` records a failed observation
for the current response, while managed accounting remains visible.
`backend_cleanup_failures` is the service-level cleanup failure count and
`backend_cleanup_latched` is `1` while further unsafe state mutation is blocked.

## Runtime states

A state summary contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Opaque `st_...` identifier. |
| `model_id`, `backend` | string | Installed model and producing backend. |
| `tier` | enum | `vram`, `ram`, `disk`, or `external`. |
| `status` | enum | `ready`, `loading`, `moving`, `unavailable`, or `quarantined`. |
| `bytes` | integer or null | Logical state size when the backend can report it. |
| `created_unix_ms`, `last_accessed_unix_ms` | integer | Unix milliseconds. |
| `expires_unix_ms` | integer or null | Policy expiry. |
| `pinned` | Boolean | Protected from policy eviction. |
| `reusable` | Boolean | Whether reuse is currently valid. |

Backend handles, filesystem paths and compatibility fingerprints are not part
of this public summary. Expiry overrides pinning: an expired pinned state is
not reusable or visible and is deleted by the next real catalog mutation.

### GET /werk/v1/states

Optional query fields are `model_id`, `tier`, `limit` and opaque `cursor`.
`limit` is 1 through the discovered `max_page_size` (currently 100). Results
are ordered by opaque state ID and return `next_cursor` when another page
exists.

~~~bash
curl -sS \
  -H "Authorization: Bearer $WERK_API_KEY" \
  'http://127.0.0.1:11434/werk/v1/states?model_id=my-model&tier=disk&limit=50'
~~~

### POST /werk/v1/states/{id}/actions

The request fields are:

~~~json
{
  "action": "promote",
  "target_tier": "ram",
  "dry_run": true,
  "allow_experimental": false
}
~~~

`action` is `pin`, `unpin`, `promote`, `demote` or `evict`.
`target_tier` is required only for promote/demote and must move in the stated
direction. `external` cannot be selected as a local movement target. The
response contains `state`, `changed` and `dry_run`; on a dry run, `state`
describes the projected result without invoking the mutation. State reads and
dry-run previews do not initialize or reconcile persistent storage, update
access metadata, quarantine corruption or delete expired entries; invalid and
expired entries are omitted or reported unavailable instead.

### POST /werk/v1/states/prune

Prune requires exactly one tagged selector. IDs:

~~~json
{
  "selector": {"kind": "ids", "ids": ["st_opaque-one", "st_opaque-two"]},
  "dry_run": true
}
~~~

Constrained filter:

~~~json
{
  "selector": {
    "kind": "filter",
    "model_id": "my-model",
    "tier": "disk",
    "older_than_unix_ms": 1788444000000
  },
  "dry_run": true
}
~~~

Every visible state:

~~~json
{
  "selector": {"kind": "all", "confirm": true},
  "dry_run": true
}
~~~

An ID selector requires 1 through `max_state_ids_per_operation` unique IDs. A
filter requires at least one non-null restriction. `all` is rejected unless
`confirm` is true. `dry_run` defaults to true if omitted. The response reports
`matched`, `removed`, nullable logical `bytes`, and `dry_run`; a preview has
`removed: 0` and does not perform catalog cleanup.

## Expert contract

### GET /werk/v1/experts

Optional query fields are `model_id`, `tier` (`vram`, `ram`, or `external`),
`limit`, `cursor`, and `allow_experimental` (default `false`). Backend routing
and its capability check are bound to one captured adapter for the operation.
A successful adapter response contains `experts` and
`next_cursor`; each expert has an opaque ID, model ID, tier, nullable byte
size, numeric hotness, pin state and optional last-use timestamp.

### POST /werk/v1/experts/actions

~~~json
{
  "model_id": "moe-model",
  "expert_ids": ["expert_opaque"],
  "action": "prefetch",
  "target_tier": "vram",
  "dry_run": true,
  "allow_experimental": false
}
~~~

Actions are `prefetch`, `pin`, `unpin` and `evict`. IDs are explicit and
bounded by `max_expert_ids_per_operation`. Target handling is strict:

| Action | `target_tier` |
| --- | --- |
| `prefetch` | Required and must be `vram` or `ram`; `external` is rejected. |
| `pin`, `unpin`, `evict` | Must be omitted. |

The response contains `experts`, `changed` and `dry_run`.

These routes define a backend-neutral contract, but no current production
adapter implements operational expert residency. Current requests therefore
fail with `unsupported`; route presence is not a support claim.

## Prefill, policy and decode

### POST /werk/v1/prefill

Text input:

~~~json
{
  "model_id": "my-model",
  "input": {"type": "text", "text": "A reusable prefix"},
  "policy": {
    "mode": "auto",
    "reuse": "prefer",
    "ttl_seconds": 3600,
    "pin": false
  },
  "allow_experimental": true
}
~~~

Message input:

~~~json
{
  "model_id": "my-model",
  "input": {
    "type": "messages",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "Describe the model."}
    ]
  },
  "policy": {"mode": "ephemeral", "reuse": "disabled", "pin": false},
  "allow_experimental": true
}
~~~

`mode` is `ephemeral`, `memory`, `disk` or `auto`; `reuse` is `disabled`,
`prefer` or `required`. The policy defaults to `auto`/`prefer`, no TTL and not
pinned. TTL is 1 through `max_ttl_seconds`. Text or total message content is
limited to 512 KiB; a message array contains 1 through 256 role/content items.

By default, omitting the top-level `policy` member uses that protocol default,
and omitting `allow_experimental` means `false`. A server started with
`werk serve --persistence` can instead provide a policy and experimental opt-in
for those two omitted members. The granular server options are
`--persistence-mode`, `--persistence-reuse`,
`--persistence-ttl-seconds` and `--persistence-pin`.

Request intent has precedence. If `policy` is present, the entire request
policy wins; this includes `{}`, whose missing inner fields receive the normal
protocol defaults rather than the server's granular defaults. An explicitly
supplied `allow_experimental` value also wins, including `false`. Server
defaults do not change capability status or bypass backend validation.

This behavior exists only at `POST /werk/v1/prefill`. It has no silent effect
on OpenAI-compatible `/v1` routes, media inference, semantic output caching,
whole-model persistence or cross-restart restore. Runtime state remains an
opaque backend-owned value governed and compatibility-checked by Werk. Current
named persistence and prefill/decode support is experimental and limited to a
functionally validated Werk-managed llama-server process for the exact
installed GGUF model; clients must still use capability discovery.

Example `data`:

~~~json
{
  "handoff": "opaque-secret-value",
  "state_id": "st_opaque-id",
  "prompt_tokens": 42,
  "reused": false,
  "tier": "disk",
  "expires_unix_ms": 1788444900000
}
~~~

`state_id` is null for ephemeral prefill. The handoff is a secret bearer value
for one decode, not a portable state serialization. Do not log it, persist it
in workflow JSON or expose it in a UI string. Live handoffs are bounded to 128
per principal and 1024 across the server. Expired handoffs are discarded first;
if a bound remains full, prefill fails with retryable `resource_exhausted`
without invalidating any live handoff. Prefill reserves its handoff slot before
backend or persistence work and rolls that reservation back on failure, so it
cannot complete expensive state work and then fail only because another
request filled the registry.

### POST /werk/v1/decode

~~~json
{
  "handoff": "opaque-secret-value",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "seed": 42,
  "stop": ["</answer>"],
  "allow_experimental": true
}
~~~

`max_tokens` is 1 through 32768. Optional temperature is 0 through 2,
optional `top_p` is greater than 0 through 1, and `stop` contains at most 16
non-empty strings of at most 1024 bytes each. Backend-returned completion text
is limited to 1 MiB of UTF-8 bytes.

Decode consumes the handoff before backend execution. It never retries on a
different backend. Consumption atomically reserves capacity for a possible
replacement handoff; that reservation is committed only if the backend returns
updated state and is otherwise released. Example `data`:

~~~json
{
  "text": "Completion text",
  "handoff": null,
  "completion_tokens": 18,
  "finish_reason": "stop"
}
~~~

`handoff` is non-null only if the backend produced a new state for a later
decode.

## Errors

| Code | HTTP status | Meaning |
| --- | --- | --- |
| `invalid_request` | 400 | The request shape, value or selector is invalid. |
| `incompatible_protocol` | 406 | The request cannot accept the server's protocol version or JSON representation. |
| `unauthorized` | 401 | Credentials are missing or invalid. |
| `forbidden` | 403 | The operation or state directory is unsafe/forbidden. |
| `not_found` | 404 | A model or state is absent or expired. |
| `conflict` | 409 | The request conflicts with current state, including required reuse with no match. |
| `incompatible_state` | 409 | The producer compatibility envelope does not match. |
| `expired_handoff` | 410 | The handoff is invalid, expired, already consumed or belongs to another principal. |
| `experimental_opt_in_required` | 428 | An experimental operation lacks explicit opt-in. |
| `resource_exhausted` | 429, or 413 for the HTTP body limit | A catalog, memory, request or other bounded resource limit was reached. |
| `unsupported` | 501 | The active adapter does not implement the operation. |
| `unavailable` | 503 | A current process, probe, telemetry or secure-identity prerequisite is missing. |
| `corrupt_state` | 500 | Stored state failed integrity validation. |
| `internal` | 500 | An internal control-plane operation failed. |

The HTTP adapter replaces internal/corrupt-state detail with a stable public
message and does not forward arbitrary backend JSON. For
`incompatible_state`, it may expose only an allow-listed
`details.mismatch_fields` string array produced by compatibility validation;
all other arbitrary detail remains redacted. `WWW-Authenticate: Bearer`
accompanies a 401 response.

## Bundled Rust client

`werk1112::werk_protocol::WerkProtocolClient` exposes typed methods matching
all ten routes:

~~~rust
use werk1112::werk_protocol::{StateListFilter, WerkProtocolClient};

let client = WerkProtocolClient::new(
    "http://127.0.0.1:11434",
    std::env::var("WERK_API_KEY").ok(),
)?;
let info = client.info()?;
let capabilities = client.capabilities()?;
let states = client.list_states(&StateListFilter::default())?;
~~~

The client requires `http://`, an explicit port and a base URL containing no
path or user information. It has a 30-second default timeout, accepts a custom
timeout, sends no redirects, bounds requests at 1 MiB and responses at 8 MiB,
and validates both content-length and chunked response framing. It sends
`Accept: application/json` and `X-Werk-Protocol-Version: 1.0`, then checks the
response header (when supplied) against the envelope. Its debug and error
representations never include the API key.

The [runtime CLI](cli.md#runtime-control) is a thin pretty-JSON interface over
this client. ComfyUI uses its own strict Python validator and keeps handoffs in
an opaque socket; see the
[custom-node guide](https://github.com/philipbodenbach/werk1112/blob/main/utils/comfyUI/README.md#runtime-persistence-experts-and-split-prefilldecode).
