# WERK native n8n nodes (Beta)

`n8n-nodes-werk1112` **1.6.0** is an **unpublished, manually installed Beta**
feature for the future Werk 1.6.0 release. Werk remains the inference/runtime
server; n8n owns workflows, item expressions, credentials and binary storage.
This package does not install Werk, models, backends, Python or media codecs.
It uses the repository's [Elastic License 2.0](LICENSE).

Core, Media Companion and ComfyUI remain at **1.5.1** in this feature branch.
Compatibility is checked through required endpoints and Werk Protocol 1.0,
not equality with the package version. All internal n8n node versions start at
**1**; saved node/operation/parameter IDs are stable contracts.

## Supported nodes

| Visible name | Operations |
| --- | --- |
| WERK Discovery (Beta) | Server info, models, one model, capabilities, complete task/model/backend parameters |
| WERK Text (Beta) | Ordered non-streaming chat messages; text, usage, finish reason and structured tool calls |
| WERK Image (Beta) | Image generation with native binary output |
| WERK Vision (Beta) | Ordered binary images in a multimodal chat request |
| WERK Video (Beta) | Text-to-video and image-to-video; submit only or submit and wait |
| WERK Audio (Beta) | Audio/music/TTS generation, audio processing and analysis |
| WERK Jobs (Beta) | Get, wait, cooperative cancel and download by output ID |
| WERK Runtime (Beta) | Info, capabilities, memory, states, state action, prune, experts, expert action, combined Prefill & Decode |

[All 30 ComfyUI registrations are mapped here](docs/comfyui-parity.md).
Text is an ordinary workflow node; it cannot connect to an AI Agent's
Chat Model socket. Returned model tool calls are data and are never executed.

## Primary manual installation

The supported validation target is **Node.js 24.12.0**, **npm 11.19.1** and
**n8n 2.37.10** (`n8n-workflow` host peer **2.37.4**). Build tooling is pinned
in `package-lock.json`, including `@n8n/node-cli` **0.46.4**. Other n8n majors
and node-loading mechanisms require separate validation.

Build from a repository checkout containing this feature using Node.js 24.
Until the feature branch is shared or merged, use the supplied local checkout;
the 1.5.1 `main` checkout does not contain this unpublished package. The clone
and branch-selection commands below apply once that branch is available:

```bash
npm install --global npm@11.19.1
git clone https://github.com/philipbodenbach/werk1112.git
cd werk1112
git switch feat/n8n-custom-nodes
cd utils/n8n
npm ci
npm run build
npm run lint
npm test
```

The npm version is intentional: npm 11.6.2 rejected its own generated
dependency lock during a clean install; the pinned newer npm is used for
the reproducible build and CI.

For an existing native n8n installation on macOS/Linux, stop n8n and copy the
**complete contents** of `dist`, including `shared`, `credentials`, `nodes`
and assets, into its own directory:

```bash
mkdir -p "$HOME/.n8n/custom/werk1112"
cp -R dist/. "$HOME/.n8n/custom/werk1112/"
```

Windows PowerShell, from `utils\n8n` after the same `npm ci` and build:

```powershell
$werkCustomDir = Join-Path $env:USERPROFILE '.n8n\custom\werk1112'
New-Item -ItemType Directory -Force -Path $werkCustomDir | Out-Null
Copy-Item -Path .\dist\* -Destination $werkCustomDir -Recurse -Force
```

Use the home directory of the account that actually runs n8n. With
`N8N_USER_FOLDER=/srv/n8n-user`, the corresponding default location is
`/srv/n8n-user/.n8n/custom/werk1112`. Do not copy only `*.node.js`: shared
modules and icons are required. The build needs no package checkout or
development `node_modules` at runtime; `n8n-workflow` is provided by n8n.

Restart n8n, search for **WERK** in the node picker, and create/test a **WERK
API** credential in one node. The custom-directory loader's actual node IDs
are recorded in [the examples](examples/README.md). A future npm community
package loader may use different IDs and would require workflow migration.
This package is not available in the npm registry, n8n Cloud or node catalog.

For development, the same compiled structure can be selected by an absolute
path. Restart n8n after rebuilding:

```bash
export N8N_CUSTOM_EXTENSIONS="$(pwd)/dist"
n8n start
```

```powershell
$env:N8N_CUSTOM_EXTENSIONS = (Resolve-Path .\dist).Path
n8n start
```

Use either this path or a copied installation for a given n8n process; loading
both creates duplicate node registrations. Existing additional extension paths
can be retained in n8n's semicolon-separated list. See the official
[custom-directory configuration](https://docs.n8n.io/deploy/host-n8n/configure-n8n/basic-configuration/configuration-examples/specify-custom-nodes-location).

For updates, stop n8n, rebuild, remove **only** the old
`custom/werk1112` directory and recopy the complete build. For uninstall,
stop n8n and remove **only** that directory (or remove this entry from
`N8N_CUSTOM_EXTENSIONS`), then restart. Existing workflows retain their node
references and must be edited before running without the package. Preserve
all other custom-node directories and n8n data.

### Existing n8n container

Add this read-only bind mount to your **existing** n8n service, preserving
its existing persistent data volume, image/version, environment and ports:

```yaml
services:
  n8n:
    volumes:
      - n8n_data:/home/node/.n8n
      - /absolute/path/to/werk1112/utils/n8n/dist:/home/node/.n8n/custom/werk1112:ro
```

This is a configuration fragment, not a standalone Compose distribution.
The container's n8n user needs read access to files and traversal access to
all directories. Restart/recreate the existing n8n service after installing
or updating. No new n8n or Werk image is needed.

When Werk runs natively on macOS and n8n runs in Docker Desktop, the
credential URL is typically `http://host.docker.internal:11434`, and Werk's
listening address/firewall must permit that connection. `localhost` inside a
container points to the container. On Linux containers, VMs, WSL or another
machine, use the actual reachable Werk host/address for that network; do not
assume Docker Desktop hostname support or automatic port reachability.

Queue workers and every other process that executes workflows need the same
compiled package, reachable Werk URL and supported shared binary storage.
Container, queue, worker and hardware inference configurations are deployment
guidance; they are not covered by the single-process mock validation.

## Credentials and network behavior

Default base URL: `http://127.0.0.1:11434`. Use HTTP locally or HTTPS behind
your TLS proxy; certificate verification is enabled. An optional proxy path
prefix is part of the base URL. Do not append `/v1`; do not put credentials,
API keys or query parameters in the URL. Authentication normally uses the
masked API-key credential field. Explicit unauthenticated mode requires a
Werk server configured for it, for example `werk serve --allow-unauthenticated`.
The credential test performs real read-only model discovery, not `/health`.

The shared HTTP helper uses n8n's credential and network checks. Werk
authentication only goes to the exact configured origin (scheme, hostname,
effective port); external output URLs never receive it. Redirects are
restricted and signed URL queries and nested credentials are removed from
errors. API keys never belong in item JSON or workflow files.

If n8n SSRF protection is enabled, narrowly allow the configured Werk host or
required addresses; for a local IPv4-only Werk endpoint, for example:

```bash
export N8N_SSRF_PROTECTION_ENABLED=true
export N8N_SSRF_ALLOWED_IP_RANGES=127.0.0.1/32
```

Choose the actual address for your deployment. Exact hostnames in
`N8N_SSRF_ALLOWED_HOSTNAMES` are another option for DNS names you control.
Hostname allowlists override address blocklists, so keep entries narrow.
See [n8n SSRF protection](https://docs.n8n.io/deploy/host-n8n/configure-n8n/security/enable-ssrf-protection).

## Models, parameters and items

Discovery joins `/v1/models.data` and `/v1/capabilities.models` by exact model
ID. Installed, declared, currently available and per-task statuses remain
distinct. Task spelling normalizes hyphens/underscores only for comparison.
Model selectors filter by task and also accept expressions/manual IDs.
An offline editor can retain existing IDs; execution validates availability
and reports server reasons. A successful probe does not promise sufficient
memory or successful model loading.

Native configuration fields cover common settings. Discovery / Parameters
returns the **complete** `/v1/parameters` schema for a task/model/backend;
the editor does not claim to generate arbitrary live controls from it.
Additional model JSON is validated, namespace-normalized and cannot overwrite
dedicated fields, routing or transport/protocol fields. List operations retain
`inherit`/`replace`/`add`/`clear` semantics. Unselected options remain absent.
Tri-state switches mean `inherit` → omitted, `enabled` → true, `disabled` →
false. Explicit zero is retained where valid; zero sample rate/channels and
TTS seed zero are specific inherited/sentinel exceptions. Seeds beyond
JavaScript's safe integer range fail instead of silently rounding.

Image `count` is distinct from model batch size. Image `response_format`
selects embedded bytes or URL delivery; `output_format` selects the file
encoding. Audio/TTS format fields follow their own endpoint contracts.
TTS forbids a negative prompt and uses asynchronous speech submission.
Vision uses ordered images in one user message and offers chat options only:
the current chat endpoint does not apply per-request media routing overrides.

Every input item evaluates its own parameters and expressions. Multiple
generated outputs use `pairedItem` to identify their source. Heavy jobs are
submitted in item order, with per-item n8n error handling; one item's bytes
or settings are never reused for another item.

### Binary and JSON outputs

Input image/audio properties contain n8n binary data, normally `data`.
Vision's image list defines the order of binary properties. Official n8n
binary helpers resolve filesystem/external references; the package never
assumes `binary.data.data` contains inline base64. A local n8n path is not a
Werk server path. The nodes transfer original bytes without transcoding.

Generated media use the binary property **`data`**, with MIME type and file
extension. JSON contains stable fields where applicable: `model`, `task`,
`text`, `jobId`, `outputId`, `status` and a sanitized `werk` object. Result
metadata preserves request/routing decisions, estimates, backend, timing and
warnings while recursively removing secrets, embedded media and internal
filesystem paths. Text/JSON/NDJSON/embedding analysis is decoded into text or
structured results, never advertised as an audio waveform.

Image `b64_json` is immediately usable binary data; any associated temporary
output ID may already be deleted. URLs and asynchronous output IDs are subject
to Werk retention. Download uses an **output ID**, never a job or result ID.

Audio task groups match ComfyUI: generation (`audio-generation`,
`music-generation`, `text-to-speech`), transforms (`voice-conversion`,
`stem-separation`, `audio-enhancement`, `audio-editing`), transcription and
translation, event/activity/speaker/language/emotion detection,
captioning/diarization/classification/understanding, and `audio-embedding`.
Generic jobs use `modality: audio`, `role: input_audio` and a base64 source
with MIME type. Reference audio is restricted to the applicable task. A
visible task choice does not imply a working backend adapter.

## Jobs, time limits and retries

HTTP 202 means the job was accepted. `queued`, `loading`, `running` and
`encoding` are pending; `completed`, `failed` and `cancelled` are terminal.
Unknown statuses, changing job IDs and completed jobs without results fail.

**Submit Only** returns immediately with the ID. Use Jobs / Get, a native
n8n Wait/If branch and Jobs / Download for durable workflow scheduling; those
operations never submit another generation. **Submit and Wait** has a finite
wait budget and polling interval. HTTP timeout, Werk inference
`timeout_seconds`, total job wait and polling interval are separate settings.
A wait timeout preserves the known job ID for later inspection. Interrupted
waiting on a job started by the node attempts best-effort cancellation;
cancellation for an already existing job is an explicit Jobs option.

DELETE requests cooperative cancellation; it does not erase job history or
files and cannot promise instant kernel interruption. Running jobs do not
resume after a Werk server restart. This package has no scheduler or global
job cache.

The client never automatically retries submission, mutations or single-use
decode after ambiguous transport failures. Enabling n8n **Retry On Fail**
reruns the node and can submit duplicate inference, repeat an action or redo
prefill/decode. Prefer submit-only plus separate status polling for long jobs.

Client limits are explicit and independent of any tighter server limits:

| Limit | Value |
| --- | --- |
| HTTP request/response wait | Default 120 seconds; maximum 3600 seconds |
| Model-option discovery / credential test | 30 seconds / 15 seconds |
| Inference JSON request/response | 128 MiB each |
| Runtime protocol request/response | 1 MiB / 8 MiB, also restricted by advertised server limits |
| Binary output | 512 MiB; embedded image decoding additionally capped at 256 MiB |
| Binary input | 64 MiB, including combined Vision images or Audio inputs |
| Structured audio-analysis output | 16 MiB |
| Job wait | Default 900 seconds; maximum 86400 seconds |
| Poll interval | Default 1 second; 0.1–300 seconds |
| Best-effort cancellation cleanup | 10 seconds |

## Runtime and persistence

Runtime operations use strict `/werk/v1` envelopes, `Accept: application/json`
and `X-Werk-Protocol-Version: 1.0`. Protocol/request IDs and DTOs are checked.
An absent response version header is accepted as in ComfyUI; a supplied
header must agree with the envelope. No runtime error falls back to `/v1`.

Capability statuses remain `supported`, `unsupported`, `unavailable`,
`experimental`, `externally_managed` or `metadata_only`. State/expert actions
are gated by those statuses and server limits. `externally_managed` permits
only the specified read-only expert telemetry. Current production adapters
do not provide operative expert residency; this integration does not turn
metadata into working MoE management.

State action, prune and expert action default to **dry-run true**. Promote
allows RAM/VRAM; demote RAM/disk; other state actions have no target tier.
Prune requires explicit IDs, real filters or explicit all plus confirmation.
It touches runtime states only. Expert actions require an explicit model and
unique expert IDs; prefetch requires RAM/VRAM and other actions forbid a tier.

**Prefill & Decode** performs both calls in one execution. Its single-use
handoff exists only in a local variable and is never returned in item JSON,
binary data, static data, logs or errors. There are no persistent separated
prefill/decode workflows in this Beta. Only safe text, state/reuse/tier/expiry
metadata and token counts leave the operation.

The current managed llama-server path requires an installed suitable GGUF
model and an explicit **Allow Experimental** opt-in. That option narrowly
permits the existing model-specific unavailable prefill probe. Capabilities
are fetched again after prefill; decode receives no unavailable exception.
The Beta display label never enables this option implicitly.

An optional enabled persistence group sends a **complete** policy
(`mode`, `reuse`, `pin`, optional positive TTL). Disabled policy is entirely
omitted so Werk applies its defaults. TTL zero in an enabled group omits TTL
and means no TTL, not immediate expiry or inheritance of a server TTL field.

## Examples, validation and troubleshooting

Import the eight [example workflows](examples/README.md), select your WERK API
credential, replace model/state IDs and supply the documented local files.
The image-to-video example includes submit-only, Wait, status branching and
download. Failed/cancelled jobs go to a visible terminal branch.

Run the real n8n loader test with a separately installed pinned host:

```bash
npx --yes npm@11.6.2 install --prefix /tmp/werk-n8n-host --save-exact n8n@2.37.10
N8N_BIN=/tmp/werk-n8n-host/node_modules/n8n/bin/n8n npm run test:loader
```

The separate n8n **test host** was installed with npm 11.6.2, which permits
the host's native dependency lifecycle scripts. Package builds use npm
11.19.1 for the reproducible lockfile and need no dependency lifecycle
scripts. CI preserves these two tested installation roles.

The loader test creates fresh temporary user directories, copies only `dist`,
starts n8n, checks real node/credential registrations and imports/executes a
workflow against a local Werk HTTP mock. Filesystem binary storage exercises
the real n8n binary helper path. It also checks the absolute custom-extension
path. No production n8n directory or GPU/model installation is involved.

[Validation scope and results](docs/validation.md) distinguish contract and
loader tests from model inference. CI installs from the package lockfile,
builds, lints, runs contracts/examples, and installs the pinned n8n host for
the loader smoke. It has read-only repository permissions and no publishing.

| Symptom | Check |
| --- | --- |
| Nodes absent | Restart n8n; correct running-user directory; complete `dist`; file permissions; avoid duplicate copied/custom-extension installs. |
| Unknown node type on import | Examples use the tested custom-directory loader IDs; npm/community loading may assign different IDs. |
| Connection fails | Base URL from the n8n process's network, Werk listener/firewall, API key/explicit unauthenticated mode, TLS proxy prefix, narrowly scoped SSRF allowlist. |
| Model unavailable | Discovery task statuses and reasons, installed model and backend, `/v1/parameters`; the node never installs missing dependencies. |
| Missing binary property | Inspect incoming binary property name; mount local test files in n8n's environment and allow access using its normal file-access settings. |
| Output download missing | Use `outputId`; retention may expire outputs; embedded image responses are already the bytes. |
| Runtime capability failure | Inspect exact capability/reason and explicit experimental opt-in. Unsupported or metadata-only is a valid result. |
| Job wait times out | Use the reported job ID with Get/Wait; increase the job budget independently of HTTP/inference limits. |

Later release preparation must synchronize component versions, review Beta
status and release notes, and separately approve any publication/tag/release.
None of those release actions is part of this integration.
