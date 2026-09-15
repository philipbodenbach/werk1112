# Beta validation scope

Release reference: `release/v1-6-0`, with Werk Core, Media Companion,
ComfyUI and the n8n package at **1.6.0**. Server contracts are based on the
actual Rust DTOs/tests and ComfyUI client in the repository. Werk Protocol
remains **1.0**, and all n8n node versions remain **1**.

## Node refresh validation (2026-09-12)

The node refresh was validated on **macOS arm64**, Node.js **24.12.0**,
package npm **11.19.1**, and a separate n8n **2.37.10** host installed with
npm **11.6.2**. A clean locked package install, build, ESLint/strict TypeScript,
**98 contract/unit/example tests**, and `npm pack --dry-run` passed.
The actual n8n process loaded all eight nodes and the credential through both
custom-directory mechanisms. Imported Image → Vision and Text → Text workflows
passed against authenticated local fixtures. The latter validates real n8n
conversation expressions, ordered history, exactly two generation POSTs and
omitted sampling/expert-cache overrides. These are API integration tests, not
an inference-speed or output-quality benchmark.

The complete ComfyUI suite passed **227 tests** with Python **3.11**, pytest
**9.1.1**, PyTorch **2.14.0**, NumPy **2.4.6**, Pillow **12.3.0** and PyAV
**18.1.0** in an isolated environment. This includes text-history round trips,
sampling inheritance, prior manual settings, and existing media/runtime tests;
the ComfyUI graphical host itself was not launched.

The failing hosted workflow had scanned only `nodes.py` and `runtime_nodes.py`,
missing the three registrations moved into `text_nodes.py`. The parity check
now discovers all mapping providers from ComfyUI's `__init__.py`, and workflow
path filters cover the full ComfyUI directory. This fixes the reproduced local
failure; no push or hosted rerun was performed by this validation.

The refresh adds item-scoped history, server-default/manual expert-cache labels,
optional inherited ComfyUI sampling, and a 600-second default HTTP timeout for
all n8n nodes. Existing stored timeouts and manual budgets remain valid.

## Environment and commands

The original integration validation used **Linux x86_64 under WSL2**, **Node.js 24.12.0**,
**npm 11.19.1**, **n8n 2.37.10**, and host peer **n8n-workflow 2.37.4**.
The development dependency versions and transitive package graph are recorded
in `package-lock.json`. The n8n host is installed separately and pinned for
the smoke test; it is not a runtime dependency inside the artifact.
That host was installed with **npm 11.6.2**, allowing its native dependency
lifecycle scripts. Package builds use **npm 11.19.1**, whose default script
restriction does not affect this package's dependency installation. CI
installs the host with Node 24.12.0's bundled npm first, then switches to
npm 11.19.1 for the locked package build.

```bash
cd utils/n8n
npm ci
npm run build
npm run lint
npm test
npm pack --dry-run
N8N_BIN=/tmp/werk-n8n-host/node_modules/n8n/bin/n8n npm run test:loader
```

The npm pin is required for the reproduced build: npm 11.6.2 generated a
dependency lock it subsequently rejected with `npm ci`; updating the tooling
and lock to npm 11.19.1 produced a successful clean install.

## Checks

The Werk **1.6.0** release-preparation pass re-ran the package checks with
Node.js **24.12.0** and npm **11.19.1**:

| Executed check | Result |
| --- | --- |
| Clean locked package install (npm 11.19.1) | Passed |
| Build, ESLint and strict TypeScript | Passed |
| Package contract/unit/example suite | 89 passed |
| `npm pack --dry-run` | Passed; complete dist, license, docs and eight example workflows |

The following results were recorded during the earlier integration
validation. The separate pinned n8n host was absent during release
preparation, so the real-loader checks were not re-run in this pass:

| Earlier integration check | Recorded result |
| --- | --- |
| Real n8n native custom-directory loader | Passed; all 8 Beta IDs and credential |
| Imported Image → Vision workflow | Passed; real expression, authentication and filesystem binary helpers |
| Absolute `N8N_CUSTOM_EXTENSIONS` loader | Passed; same node IDs |
| Eight workflows checked against live type manifests | Passed |
| Existing ComfyUI regression | 190 passed |

CI is configured to reproduce these commands; no hosted CI run or release
action was triggered as part of the local implementation.

- **Build and package contracts:** private unpublished package, Elastic-2.0
  license, eight Beta node definitions at version 1, one masked credential,
  icons/shared JavaScript present in the complete dist structure, no runtime
  dependency installation or release/publish scripts.
- **Transport/discovery/jobs:** controlled responses test authentication,
  origin and redirects, nested redaction, model/task distinctions, parameters,
  limits/timeouts, all job statuses, cancellation and no duplicate submission.
- **Media:** strict request builders, safe integers, inheritance/false/zero,
  namespaces/list operations, real binary-helper contracts and simulated
  external references, expressions across items, paired outputs, ordered
  vision inputs, asynchronous TTS and structured audio artifacts.
- **Runtime:** strict envelopes/version headers/DTOs, exact capability gates,
  server bounds, dry-run defaults, explicit prune/expert selectors, policy
  omission, experimental prefill probe, capability recheck and private
  single-use handoff handling.
- **Examples/parity:** every current public ComfyUI registration has exactly one
  table row (33, including the three Text nodes); eight workflow files contain no credentials and use real node
  versions, parameter IDs and graph references.
- **Existing ComfyUI regression:** the earlier integration validation
  recorded 190 passing tests. Repository-wide release checks are separate
  from this n8n package validation.

The loader smoke is a distinct integration check. It copies only `dist`
into a newly created temporary `.n8n/custom/werk1112`, starts the actual n8n
binary, creates a local fixture owner/session and reads the authenticated
type manifests. It verifies these real names and their credential:

```text
CUSTOM.werkDiscovery  CUSTOM.werkText   CUSTOM.werkImage
CUSTOM.werkVision     CUSTOM.werkVideo  CUSTOM.werkAudio
CUSTOM.werkJobs       CUSTOM.werkRuntime
credential: werkApi
```

The smoke checks all example parameters against the live manifests, imports
fixture credentials and a workflow using n8n's CLI, and executes Image →
Vision against a local authenticated HTTP mock behind a `/proxy` path prefix.
An item-index expression runs through n8n's actual expression engine and the
mock checks the resolved prompt.
Filesystem binary mode requires the image output to have a real binary-store
ID and the vision HTTP input to contain exactly the original bytes. A
second fresh user directory verifies the absolute `N8N_CUSTOM_EXTENSIONS`
path with the same complete artifact. The fixture explicitly enables n8n's
SSRF protection and permits only `127.0.0.1/32`. No checkout dependency path,
production database, user credential or installed model is used.

## Real limits of this evidence

These are contract, loader and workflow/binary-transfer tests, **not real
model inference**. No GPU, actual media backend, GGUF prefill/decode,
production MoE expert management, Windows n8n process, n8n container,
queue worker or external binary-storage service was exercised. External
binary references are simulated in unit tests; real filesystem storage is
exercised by n8n itself.

The pinned host emits a notice that native deployment is deprecated for
future n8n versions. This Beta validates the currently tested 2.37.10 custom
loader only; it does not promise future native-loader compatibility. Its
optional internal Python-runner warning does not affect this workflow:
none of the WERK nodes uses a Python or Code node.

Capability discovery can truthfully report unsupported, unavailable,
externally managed or metadata-only. The package does not install missing
adapters or invent a successful execution. See [contract clarifications and
the text-readiness distinction](comfyui-parity.md#contract-clarifications-checked-in-source).
The package remains private and manually installed for Werk **1.6.0**.
No npm publication, Registry submission, cloud verification, Docker image,
tag or release was performed by these validation commands.
