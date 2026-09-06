# Importable WERK workflows (Beta)

Import a JSON file in n8n, select your **WERK API** credential on every WERK
node, and replace each `REPLACE_*` model/state ID with one from your server's
Discovery/Runtime results. No credentials, private paths or model downloads are
embedded. Examples are inactive and use node version 1.

The primary **complete-dist custom-directory** installation loads these IDs
in n8n **2.37.10**:

```text
CUSTOM.werkDiscovery   CUSTOM.werkText    CUSTOM.werkImage
CUSTOM.werkVision      CUSTOM.werkVideo   CUSTOM.werkAudio
CUSTOM.werkJobs        CUSTOM.werkRuntime
```

The real loader smoke checks these IDs against `/types/nodes.json` from a
started n8n process and validates every example's node/parameter references.
These are not assumed npm-community-package IDs. Switching to a future npm
installation can require workflow migration.

| File | Prerequisites and result |
| --- | --- |
| [01-discovery-text.json](01-discovery-text.json) | Installed text model; discovery followed by a short normal chat completion. |
| [02-image.json](02-image.json) | Available image-generation model; inspect generated binary `data` in n8n. No GPU/model is bundled. |
| [03-vision.json](03-vision.json) | Suitable vision model and readable `/data/werk-examples/input.png`; ordered binary input, text analysis. Change the read node to your own local file path. |
| [04-image-to-video-jobs.json](04-image-to-video-jobs.json) | Image-to-video model and the same image; submit once, native Wait, Get, completed/terminal branching and output-ID download. The example downloads the first output (`count: 1`). Failed/cancelled jobs stop on an inspectable branch. Stop the workflow to end polling of a job that never becomes terminal. |
| [05-tts.json](05-tts.json) | Suitable TTS model; asynchronous speech job, finite wait and binary audio `data`. |
| [06-audio-analysis.json](06-audio-analysis.json) | Speech-to-text model and readable `/data/werk-examples/input.wav`; structured transcription results. Change the file path and, if desired, choose another supported analysis task/model. |
| [07-runtime-dry-run.json](07-runtime-dry-run.json) | Protocol 1.0 server and explicit visible state ID; runtime info and dry-run pin. Enable experimental opt-in only if the server reports that action as experimental. Without a matching capability/state, the action intentionally fails. |
| [08-prefill-decode.json](08-prefill-decode.json) | Installed GGUF on an eligible managed llama-server backend. **Allow Experimental is explicitly true in this example**; review before running. Handoff remains private inside one execution. Policy is explicitly memory/prefer/no TTL/not pinned. |

The file paths above are replaceable examples in **n8n's** filesystem, not
Werk server paths. On Windows select a local readable Windows path; in an
existing container mount the directory into n8n. Respect n8n's own configured
file-access restrictions. Files and model IDs are prerequisites, not supplied
fixtures for real model inference.

Model-specific dimensions, frame limits and optional parameters should come
from Discovery / Parameters and the model documentation. Examples leave most
model options absent so server/model defaults remain authoritative. Outputs
expire under Werk retention. Download by output ID before expiry, then use
n8n's file/storage nodes to retain bytes as appropriate.
