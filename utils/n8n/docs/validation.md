# Beta validation scope

Reference checkout: `82629bfc6d1dfc8c8fac2ccba604c987f13059ae`, Werk Core,
Media Companion and ComfyUI **1.5.1**, with the new n8n package targeting
**1.6.0**. Existing server routes and component version files are unchanged.
Server contracts were checked against the actual Rust DTOs/tests and the
ComfyUI 1.5.1 client, including Werk Protocol **1.0**.

## Environment and commands

Validation uses **Linux x86_64 under WSL2**, **Node.js 24.12.0**,
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
N8N_BIN=/tmp/werk-n8n-host/node_modules/n8n/bin/n8n npm run test:loader
```

The npm pin is required for the reproduced build: npm 11.6.2 generated a
dependency lock it subsequently rejected with `npm ci`; updating the tooling
and lock to npm 11.19.1 produced a successful clean install.

## Checks

| Executed check | Result |
| --- | --- |
| Clean locked package install (npm 11.19.1) | Passed |
| Build, ESLint and strict TypeScript | Passed |
| Package contract/unit/example suite | 89 passed |
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
- **Examples/parity:** every public ComfyUI registration has exactly one table
  row (30); eight workflow files contain no credentials and use real node
  versions, parameter IDs and graph references.
- **Existing ComfyUI regression:** 190 tests passed; no server modification
  required an additional Rust test matrix.

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
production MoE expert management, macOS/Windows n8n process, n8n container,
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
No npm publication, Registry submission, cloud verification, Docker image,
tag, release or component-version synchronization was performed.
