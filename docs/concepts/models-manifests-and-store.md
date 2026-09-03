# Models, manifests and the managed store

Werk copies models into a managed store and records a local manifest for every
installed model. Importing or classifying a repository does not imply that a
runtime can execute it. Use `werk doctor --model MODEL --task TASK --debug`
before a large first load.

See [Tasks and formats](../reference/tasks-and-formats.md) for the canonical
task, modality, layout and format values, and [Backends](../backends.md) for
runtime selection.

## Store root

The global `--model-home PATH` option has highest priority. Without it, the
store root is resolved in this order:

1. `WERK_HOME`;
2. `XDG_DATA_HOME/werk1112`;
3. on Windows, `LOCALAPPDATA/werk1112`;
4. on Windows without `LOCALAPPDATA`, `USERPROFILE/AppData/Local/werk1112`;
5. otherwise `HOME/.local/share/werk1112`.

The main layout is:

~~~text
WERK_HOME/
├── models/
│   └── MODEL_DIRECTORY/
│       ├── manifest.json
│       └── files/
├── artifacts/
│   └── MODEL_DIRECTORY/
│       └── onnx/
├── backends/
├── outputs/
├── jobs/
├── auth/
│   └── huggingface.token
└── tmp/
~~~

Backend installers own children below `backends/`; models, optimized artifacts,
outputs and jobs are separate siblings. See
[Managed backend locations](../backends.md#managed-locations) before removing
anything manually.

Model IDs remain as entered in `manifest.json`. Directory names retain ASCII
letters, digits, `.`, `_` and `-`; other characters are replaced with `-`.
Empty IDs, IDs beginning with `-`, and IDs containing `..` are rejected.

## Import, pull and removal

### Local import

~~~bash
werk import /path/to/model-or-directory --name local-model
~~~

Werk copies a file, or the contents of a directory, into
`models/LOCAL-MODEL/files/`. It does not leave the model pointing at the
original path. A source directory's `.git` directory is not copied. Import
fails if the target model directory already exists.

### Hugging Face pull

~~~bash
werk pull org/repository --name model-name
werk pull org/repository --name model-name --file model.Q4_K_M.gguf
~~~

Pull performs a shallow Git clone, then resolves Git LFS content before copying
the result into the managed store. `git` and `git-lfs` must already be
installed. A selected `--file` must be a relative path inside the repository;
Werk copies that file plus recognized non-weight repository metadata needed for
classification and tokenization.

When no `--file` is supplied and a repository is detected as GGUF, Werk chooses
one GGUF file for the LFS pull. Its current filename preference starts with
`Q4_K_M`, then `Q5_K_M`, `Q4_K_S`, `Q5_K_S`, `Q6_K`, `Q8_0`, the Q3 variants,
`Q4_0`, `Q5_0`, and `Q2_K`; lexical order breaks ties. Pass `--file` when that
policy is not the intended quantization.

Hugging Face authentication is resolved from `HF_TOKEN`, then
`HUGGING_FACE_HUB_TOKEN`, then the Werk token under `auth/`, then the normal
Hugging Face CLI token file. Accepting gated-model terms still requires the
Hugging Face website.

### Removal

~~~bash
werk remove model-name
~~~

Removal deletes the model directory and its sibling `artifacts/MODEL_DIRECTORY`
tree. It does not remove outputs, jobs, other models or managed backends.

## Manifest identity and inventory

Each `manifest.json` contains these stable identity and inventory fields:

| Field | Meaning |
| --- | --- |
| `id` | Installed model ID. |
| `source` | Tagged `local_path` or `hugging_face` source record. |
| `format` | Detected model format. |
| `architecture` | GGUF `general.architecture` or Transformers-style configuration identity when detectable. |
| `tokenizer_path` | Tracked relative `tokenizer.json` path, if present. |
| `config_path` | Tracked relative `config.json` path, if present. |
| `model_path` | Selected primary weight path; Diffusers repositories normally use the repository root and therefore have no component `model_path`. |
| `backend` | Human-facing backend hint, not a runtime guarantee. |
| `created_unix` | Creation timestamp in Unix seconds. |
| `files` | Relative path, byte size and `crc32:` checksum for each copied file. |
| `artifacts` | Persisted optimized-artifact records. Currently the concrete artifact kind is ONNX. |

The CRC32 inventory detects accidental file changes; it is not a cryptographic
content identity or authenticity proof.

## Schema-v2 capability metadata

Schema-v2 metadata is flattened into the same JSON object:

| Field | Meaning |
| --- | --- |
| `schema_version` | Current persisted schema is `2`; absent legacy fields deserialize as schema `1`. |
| `family` | Normalized model-family hint. |
| `repository_layout` | `single_file`, `gguf`, `transformers`, `diffusers`, `mlx`, `onnx_bundle`, `tensorrt_engine` or `custom`. |
| `tasks` | Canonical task list. |
| `input_modalities`, `output_modalities` | Modalities derived from or curated with the tasks. |
| `components` | Typed component paths plus optional format, precision, quantization and file lists. |
| `precision`, `quantization` | Detected model-level hints. |
| `generation_defaults` | Values copied from recognized generation/config metadata. |
| `parameter_constraints` | Detected model capability hints such as dimensions or sequence limits. Reported maxima are visible to clients and warn when exceeded, but do not silently clamp or reject an explicit override before the selected backend sees it. |
| `compatible_runtimes` | Planner hints derived from the current manifest; availability is checked separately. |
| `optimized_artifacts` | Flattened summary of artifact kind, path and status. |
| `chat_template` | Resolved GGUF/model chat-template description when one can be established without guessing. |

Detection uses local repository evidence such as `config.json`,
`model_index.json`, `generation_config.json`, GGUF metadata, filenames and
component directories. Task/family detection is intentionally conservative but
still heuristic. A runtime probe and first model load remain authoritative.

Legacy manifests are enriched in memory when read. Merely running `werk list`
or `werk inspect` does not promise to rewrite the on-disk file. Commands that
intentionally change manifest state, such as `werk select-file` or artifact
building, persist their result.

## Inspecting and filtering

~~~bash
werk list
werk list --task text-to-speech
werk list --input-modality audio --output-modality text
werk list --family whisper --layout transformers
werk list --json
werk inspect model-name
~~~

`werk list` reads installed manifests, enriches them from local files and sorts
them by model ID. Filters cover task, input modality, output modality, family,
layout and an explicitly selected global backend.

`werk inspect` prints the enriched manifest as JSON and adds a dynamic
`host_resources` object to that command's output. `host_resources` is not part
of the persisted manifest.

## Selecting a tracked weight

Repositories can contain several weights or quantizations:

~~~bash
werk inspect model-name
werk select-file model-name model.Q5_K_M.gguf
~~~

The selected file must already appear in the manifest inventory and must stay
inside the installed model tree. Werk updates `format`, `model_path`,
`architecture` and the backend hint, then re-runs capability enrichment.
Metadata that differs from inference-derived values is preserved as a curated
override.

## Optimized artifacts

~~~bash
werk artifacts build model-name
werk artifacts list model-name
werk artifacts rebuild model-name
~~~

Current artifact building exports supported safetensors text architectures to
`artifacts/MODEL_DIRECTORY/onnx/`. The exporter is discovered from
`WERK_ONNX_EXPORTER`, `optimum-cli`, or a Python interpreter capable of running
the Optimum ONNX module. Artifact records are also written back to the model
manifest. A failed export is retained with status `failed` and diagnostic
detail.

## Offline execution boundary

Import and pull are the workflows that place weights in the store. The media
companion itself forces Hugging Face, Transformers, Diffusers and datasets
offline modes and passes local-only loading options. It does not install
packages or download missing weights during inference.

## Related reference

- [Tasks and formats](../reference/tasks-and-formats.md)
- [Environment variables](../reference/environment-variables.md)
- [Backends, routing and platform support](../backends.md)
- [Media inference](../media-inference.md)

Implementation source: [`src/model_store.rs`](https://github.com/philipbodenbach/werk1112/blob/main/src/model_store.rs),
[`src/capabilities`](https://github.com/philipbodenbach/werk1112/tree/main/src/capabilities), and
[`src/cli.rs`](https://github.com/philipbodenbach/werk1112/blob/main/src/cli.rs).
