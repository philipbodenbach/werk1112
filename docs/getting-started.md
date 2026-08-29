# Getting started

This guide installs the Werk1112 binary, adds a model to the managed store and
runs a first local request. Models, accelerator drivers and optional inference
runtimes are installed separately.

## Install a release binary

Linux and macOS:

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.sh)"
```

The shell installer selects `linux-x86_64`,
`linux-x86_64-amd-strix-halo`, `linux-aarch64-dgx-spark`, or
`macos-aarch64` from the current host. On Linux x86_64 it selects the Strix
Halo profile only when specific CPU, DMI, Radeon 8050S/8060S/8040S, or
`gfx1151` ROCm signals identify the host; other x86_64 systems keep the generic
artifact. The ARM64
artifact is accepted only when `/proc/device-tree/model` or `nvidia-smi`
identifies DGX Spark/GB10; other Linux ARM64 hosts must build from source.
The installer verifies the published `.sha256` file before extracting any
release archive. Models and accelerator runtimes remain separate from the Werk
binary. See the [Strix Halo](integrations/strix-halo.md) and
[DGX Spark](integrations/dgx-spark.md) platform guides.

The default destination is `$HOME/.local/bin`. Select a release or destination
with environment variables:

```bash
WERK_VERSION=1.4.0 \
WERK_INSTALL_DIR="$HOME/bin" \
sh -c "$(curl -fsSL https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.sh)"
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.ps1 | iex
```

The default Windows destination is
`%LOCALAPPDATA%\Programs\Werk1112\bin`. Select a release, destination or PATH
update before invoking the installer:

```powershell
$env:WERK_VERSION = "1.4.0"
$env:WERK_INSTALL_DIR = "$HOME\bin"
$env:WERK_ADD_TO_PATH = "1"
irm https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/install.ps1 | iex
```

Verify the installed command:

```bash
werk --version
werk --help
```

## Install from source

A source install requires the stable Rust toolchain and the native build
dependencies of the selected Cargo features:

```bash
cargo +stable install --path . --locked
```

For a portable CPU-only development build:

```bash
cargo +stable build --locked --no-default-features
```

Target release builds and their platform prerequisites are documented in
[Building from source](development/build.md).

## Add a model

Import copies a local file or directory into Werk's managed store:

```bash
werk import /absolute/path/to/model --name local-model
```

Pull downloads a Hugging Face repository. Real model repositories require Git
and Git LFS:

```bash
git lfs install
werk pull organization/repository --name model-name
```

Repositories containing many GGUF quantizations can be limited to one file:

```bash
werk pull organization/gguf-repository \
  --file model.Q4_K_M.gguf \
  --name model-q4
```

For gated repositories, accept the conditions on the model page and then use
one of the supported authentication sources:

```bash
werk auth huggingface login
werk auth huggingface status
```

Werk also recognizes `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` and the standard
Hugging Face CLI token cache. It cannot accept repository conditions for the
user.

Inspect what was installed before inference:

```bash
werk list
werk inspect model-name
werk doctor --model model-name --debug
```

An imported model can be catalogued even if no currently installed backend can
execute its architecture. `doctor` and the first cold model load distinguish
those cases.

## Run inference

Text chat:

```bash
werk chat local-model --max-tokens 128
```

Typed media:

```bash
werk image generate IMAGE_MODEL --prompt "A quiet orbital greenhouse"
werk video generate VIDEO_MODEL --prompt "Clouds crossing a mountain ridge"
werk audio generate speech TTS_MODEL \
  --text "Werk elf zwölf ist bereit." \
  --output speech.wav
```

Use `--verbose` for measured timings and `--debug` for request resolution,
candidate rejection reasons and the selected runtime:

```bash
werk audio generate speech TTS_MODEL \
  --text "Routing diagnostic." \
  --backend auto --verbose --debug
```

See [Media inference](media-inference.md) for canonical tasks and parameters,
and [Backends](backends.md) for runtime installation and fallback behavior.

## Start the HTTP service

Authentication is enabled by default. Generate a key and start the server:

```bash
werk auth api-key generate
export WERK_API_KEY="replace-with-generated-key"
werk serve --model local-model
```

OpenAI-compatible clients use this base URL:

```text
http://127.0.0.1:11434/v1
```

For a deliberately unauthenticated loopback development server:

```bash
werk serve --model local-model --allow-unauthenticated
```

Do not expose an unauthenticated server on a public interface. The complete
route, authentication and response contract is in the [HTTP API](api.md).

## Uninstall Werk

Linux and macOS:

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/uninstall.sh)"
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/philipbodenbach/werk1112/main/scripts/uninstall.ps1 | iex
```

The uninstallers remove the binary from the same default or
`WERK_INSTALL_DIR` location. If a model store or API-key file is found, the
scripts ask separately before deleting it. The default answer keeps user data.

Managed backend environments have a separate lifecycle. The current cleanup
procedure and the absence of a `werk backend uninstall` command are documented
under [Backend uninstall and cleanup](backends.md#uninstall-and-cleanup).
