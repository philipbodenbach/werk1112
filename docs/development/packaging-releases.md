# Packaging and releasing Werk1112

Werk1112's release tooling produces one router artifact for each configured
platform profile. It does not produce a separate archive for every backend.
Runtime availability is determined later from compiled Werk features, managed
backend installations, host-installed runtimes and configured remote services.

This page follows the current packaging scripts:

- [`scripts/package-release.sh`](https://github.com/philipbodenbach/werk1112/blob/main/scripts/package-release.sh) for the
  shell packaging surface;
- [`scripts/package-release.ps1`](https://github.com/philipbodenbach/werk1112/blob/main/scripts/package-release.ps1) for native
  Windows packaging;
- [`scripts/build-windows.ps1`](https://github.com/philipbodenbach/werk1112/blob/main/scripts/build-windows.ps1) for the guarded
  Windows build;
- [`.cargo/config.toml`](https://github.com/philipbodenbach/werk1112/blob/main/.cargo/config.toml) for target aliases;
- [`Cargo.toml`](https://github.com/philipbodenbach/werk1112/blob/main/Cargo.toml) for the version and release feature bundles.

See [Building from source](build.md) for toolchain setup and feature details,
and [Backend support](../backends.md) for runtime provisioning after release
installation.

## Configured target matrix

| Platform | Cargo alias | Binary | Release archive |
| --- | --- | --- | --- |
| Linux x86_64 | `cargo +stable build-linux` | `target/x86_64-unknown-linux-gnu/release/werk` | `werk1112-v<VERSION>-linux-x86_64.tar.gz` |
| Linux x86_64 / AMD Strix Halo | `cargo +stable build-linux-strix-halo` | `target/x86_64-unknown-linux-gnu/release/werk` | `werk1112-v<VERSION>-linux-x86_64-amd-strix-halo.tar.gz` |
| Linux aarch64 / DGX Spark | `cargo +stable build-linux-aarch64` | `target/aarch64-unknown-linux-gnu/release/werk` | `werk1112-v<VERSION>-linux-aarch64-dgx-spark.tar.gz` |
| Windows 10/11 x64 | `cargo +stable build-windows` or `scripts/build-windows.ps1` | `target/x86_64-pc-windows-msvc/release/werk.exe` | `werk1112-v<VERSION>-windows-x86_64.zip` |
| macOS Apple Silicon | `cargo +stable build-macos-apple-silicon` | `target/aarch64-apple-darwin/release/werk` | `werk1112-v<VERSION>-macos-aarch64.tar.gz` |

Windows arm64, macOS x86_64 and other target combinations are not currently
produced by the checked-in release scripts.

The version embedded in each archive name is read from the `[package]` version
in `Cargo.toml`. The scripts write archives under `releases/` and a sibling
`.sha256` file for each archive.

## What an archive contains

The current scripts stage exactly these files:

| Archive | Files |
| --- | --- |
| Linux/macOS tarball | `werk`, `README.md`, `LICENSE` |
| Windows zip | `werk.exe`, `README.md`, `LICENSE` |

The media companion implementation needed by the binary is embedded at build
time. The archive does **not** contain:

- model weights or optimized model artifacts;
- CUDA, ROCm, Metal, Vulkan or accelerator drivers/toolkits;
- llama.cpp, vLLM, ONNX Runtime, MLX, oMLX or managed Python environments;
- Diffusers, Transformers, audio/video codecs or other optional Python
  packages;
- Rust, Cargo, Visual Studio, CMake, Git, libclang or `nvcc`;
- the full `docs/` tree.

The included `LICENSE` contains the authoritative terms for Werk1112. It does
not relicense third-party dependencies, models, runtimes, or other materials;
those retain their own licenses.

## Build on the matching platform

Each packaging command invokes its target build before creating the archive.
The configured Rust target alone does not provide a foreign linker, SDK or
accelerator toolchain. Unless a complete cross-compilation environment exists:

- package Linux x86_64 from native x86_64 Linux or WSL;
- package the Strix Halo profile natively on AMD Ryzen AI Max/`gfx1151`;
- package Linux aarch64 natively on DGX Spark/GB10;
- package Windows from native Windows Developer PowerShell;
- package macOS from Apple Silicon macOS.

Do not use WSL to produce the native Windows archive. WSL can produce the Linux
artifact.

## Linux package

From the repository root on a Linux host with the release CUDA toolchain:

~~~bash
./scripts/package-release.sh linux
~~~

The shell script:

1. reads the version from `Cargo.toml`;
2. runs `cargo build-linux`;
3. verifies the expected target binary;
4. recreates `target/package/linux` as a staging directory;
5. copies `werk`, the current root `README.md`, and the authoritative
   `LICENSE`;
6. writes the gzip-compressed tar archive;
7. writes a SHA-256 checksum with `sha256sum`, or `shasum -a 256` when
   `sha256sum` is unavailable.

It requires `tar` in addition to the build prerequisites.

## AMD Strix Halo / Linux x86_64 package

Run the Strix Halo packaging branch natively on a Ryzen AI Max release host:

~~~bash
./scripts/package-release.sh linux-strix-halo
~~~

It invokes `cargo build-linux-strix-halo`, selects the backend-neutral
`release-linux-strix-halo` feature, stages the normal three files and writes:

~~~text
releases/werk1112-v<VERSION>-linux-x86_64-amd-strix-halo.tar.gz
releases/werk1112-v<VERSION>-linux-x86_64-amd-strix-halo.tar.gz.sha256
~~~

Before invoking Cargo, the packager requires native Linux x86_64 plus a
specific Ryzen AI Max/Strix Halo signal from CPU or DMI identity, a matching
Radeon 8050S/8060S/8040S identity, or a `gfx1151` agent from
`rocm_agent_enumerator`/`rocminfo`. A generic x86_64 builder cannot
label its output as Strix Halo merely because the Rust target matches. The
gate verifies host identity, not working ROCm kernels; complete the
[hardware smoke gate](../integrations/strix-halo.md#hardware-release-smoke-gate)
before publishing.

## DGX Spark / Linux aarch64 package

Run the aarch64 packaging branch natively on the DGX Spark release builder:

~~~bash
./scripts/package-release.sh linux-aarch64
~~~

It invokes `cargo build-linux-aarch64`, which selects
`aarch64-unknown-linux-gnu`, the `release-linux-aarch64` CUDA bundle and compute
capability 12.1 (`sm_121`). It then stages the same three files as the x86_64
Linux package and writes:

~~~text
releases/werk1112-v<VERSION>-linux-aarch64-dgx-spark.tar.gz
releases/werk1112-v<VERSION>-linux-aarch64-dgx-spark.tar.gz.sha256
~~~

The checked-in packaging command intentionally requires a native Spark/GB10
build host. Before invoking Cargo, the packager requires Linux aarch64 plus a
DGX Spark/GB10 signal from `/proc/device-tree/model` or `nvidia-smi`. It does
not attempt to synthesize an ARM64 sysroot or CUDA cross-toolchain on x86_64.
The resulting archive still requires a separate native smoke test before
release.

## macOS package

From the repository root on Apple Silicon macOS:

~~~bash
./scripts/package-release.sh macos
~~~

The flow is the same as Linux but invokes
`cargo build-macos-apple-silicon` and stages the arm64 binary. It requires
`tar` and either `shasum` or `sha256sum`.

## Native Windows package

Use the PowerShell packager from native x64 Developer PowerShell:

~~~powershell
.\scripts\package-release.ps1 -Target windows
~~~

`windows` is currently the only accepted PowerShell target and is also the
default. The script:

1. reads the version from `Cargo.toml`;
2. invokes `scripts/build-windows.ps1`;
3. verifies `target\x86_64-pc-windows-msvc\release\werk.exe`;
4. recreates `target\package\windows`;
5. copies `werk.exe`, `README.md`, and `LICENSE`;
6. creates the zip with `Compress-Archive`;
7. writes a lowercase SHA-256 checksum using `Get-FileHash`.

The nested build script rejects WSL/non-Windows execution and checks for the
x64 MSVC compiler and CUDA compiler. See the
[native Windows build instructions](build.md#native-windows-x64-release-build)
before packaging.

The shell packager also accepts a `windows` argument and uses `zip`, but it
still invokes the Windows Cargo alias. The native PowerShell entry point is the
documented path for normal Windows releases.

There is intentionally no aggregate `all` mode. Strix Halo and DGX Spark
artifacts require mutually exclusive native hardware identities, while the
Windows and macOS artifacts require their own platform SDKs. Build each
artifact on its matching host and aggregate the checksum-verified files in the
release workflow.

## Local output layout

For package version `1.7.0`, the generated tree is:

~~~text
releases/
├── werk1112-v1.7.0-linux-x86_64.tar.gz
├── werk1112-v1.7.0-linux-x86_64.tar.gz.sha256
├── werk1112-v1.7.0-linux-x86_64-amd-strix-halo.tar.gz
├── werk1112-v1.7.0-linux-x86_64-amd-strix-halo.tar.gz.sha256
├── werk1112-v1.7.0-linux-aarch64-dgx-spark.tar.gz
├── werk1112-v1.7.0-linux-aarch64-dgx-spark.tar.gz.sha256
├── werk1112-v1.7.0-windows-x86_64.zip
├── werk1112-v1.7.0-windows-x86_64.zip.sha256
├── werk1112-v1.7.0-macos-aarch64.tar.gz
└── werk1112-v1.7.0-macos-aarch64.tar.gz.sha256
~~~

Staging directories are recreated below `target/package/<platform>`. Existing
archive and checksum files with the same version and platform name are
replaced by the packaging script.

## Verify an artifact

Inspect Unix archive contents:

~~~bash
tar -tzf releases/werk1112-v<VERSION>-linux-x86_64.tar.gz
tar -tzf releases/werk1112-v<VERSION>-linux-x86_64-amd-strix-halo.tar.gz
tar -tzf releases/werk1112-v<VERSION>-linux-aarch64-dgx-spark.tar.gz
tar -tzf releases/werk1112-v<VERSION>-macos-aarch64.tar.gz
~~~

Inspect the Windows archive on a Unix host with `unzip`:

~~~bash
unzip -l releases/werk1112-v<VERSION>-windows-x86_64.zip
~~~

Verify a shell-generated checksum from inside `releases/`:

~~~bash
cd releases
sha256sum -c werk1112-v<VERSION>-linux-x86_64.tar.gz.sha256
~~~

On Windows, compare the generated file with:

~~~powershell
Get-Content .\releases\werk1112-v<VERSION>-windows-x86_64.zip.sha256
Get-FileHash .\releases\werk1112-v<VERSION>-windows-x86_64.zip -Algorithm SHA256
~~~

After extracting on the matching target, run the packaged binary directly:

~~~bash
./werk --help
~~~

~~~powershell
.\werk.exe --help
~~~

Artifact verification should test the router binary itself. Optional runtime
health remains host-specific and is checked after installation with
`werk backend doctor --debug`.

## Relationship to end-user installers

The installer scripts download these exact artifact names from a GitHub release
whose tag is `v<VERSION>`:

| Installer | Supported downloads |
| --- | --- |
| `scripts/install.sh` | `linux-x86_64` on generic Linux x86_64; `linux-x86_64-amd-strix-halo` when specific CPU, DMI, Radeon 8050S/8060S/8040S, or `gfx1151` signals identify Strix Halo; `linux-aarch64-dgx-spark` only when Linux arm64 identifies DGX Spark/GB10; `macos-aarch64` on Apple Silicon macOS |
| `scripts/install.ps1` | `windows-x86_64` on native Windows |

Both installers accept `WERK_VERSION` with or without a leading `v`. When it is
unset they query the latest GitHub release. `WERK_REPO` can select a different
GitHub repository, and `WERK_INSTALL_DIR` changes the binary destination.
The POSIX installer downloads the archive's sibling `.sha256`, requires
`sha256sum` or `shasum`, verifies it before extraction, and rejects archives
whose entries are not exactly `werk`, `README.md`, and `LICENSE`.

The packaging scripts only create local files. They do not create a Git tag,
create a GitHub release or upload artifacts.

## Manual GitHub release workflow

Prepare a release pull request (for example, `release/v1-7-0`) with the intended
version in all product version files, a dated changelog section and comparison
links, README highlights and updated installation examples. Complete validation
and merge the PR into the default branch before starting the release workflow.

Open **Actions → Release → Run workflow** and select the default branch.
The workflow reads the already prepared version from `Cargo.toml`; it does not
increment versions or create another commit. It checks that Cargo.lock, Media
Companion, ComfyUI and both n8n metadata files agree, that the version is newer
than the latest stable release tag, and that its dated changelog entry contains
release notes and correct comparison links. The tag must not already exist.

The workflow creates an annotated `v<VERSION>` tag on the exact commit selected
when the run was dispatched, then creates a GitHub release with the prepared
changelog entry as its notes. Repository files and the default branch are not
modified by the workflow. The workflow must be present on the default branch
before it appears in GitHub's manual workflow menu.

**Keep release as a draft** is enabled by default. Build the platform artifacts
from the generated tag, upload the archives and their `.sha256` files to the
draft, then publish it. This keeps the previous release available to installers
until the new downloads are ready. Disable the draft option only when you want
to publish immediately, before manually attaching artifacts.

The workflow runs release-tool tests and metadata checks. It does not build
binaries, upload assets, run the full product test suites, publish to npm, or
publish to the ComfyUI Registry. Protocol, dependency and node schema versions
follow their own compatibility rules.

The built-in `GITHUB_TOKEN` supplies `contents: write`; no additional secret is
needed. Repository rules must permit it to create release tags. The workflow
does not push to the protected default branch or bypass tag protection rules.

If the tag was pushed but GitHub release creation failed, recover the existing
tag through GitHub's release UI or `gh release create v<VERSION> --verify-tag
--draft --title "Werk1112 v<VERSION>" --notes-file release-notes.md`, using the
notes from its dated changelog entry. Do not move or delete the tag.

The local helper prepares the release PR files from a clean checkout whose
current version matches the latest stable tag. Maintain the changes under
`Unreleased` in `CHANGELOG.md`, including a short `### Highlights` section:

~~~markdown
## [Unreleased]

### Highlights

- Describe a user-facing change and its relevant limitations.

### Fixed

- Describe other fixes that belong in the full release notes.
~~~

Commit these notes on the release branch, then run:

~~~bash
python3 scripts/prepare-release.py minor --notes-file /tmp/werk-release-notes.md
~~~

Use `patch`, `minor` or `major` as appropriate. The helper synchronizes all
product versions and lockfiles, creates the dated changelog section, and
rebuilds the README's "What's new" section from those highlights. It also
updates the current release references in installation instructions, artifact
names, protocol examples, integration READMEs, parity documentation and the
validation page's release reference. The rest of the README, historical
changelog entries and recorded validation results remain intact.

Review and commit the generated files in the release PR, then merge it.
Content-specific API documentation and fresh validation evidence still need
to be maintained with the corresponding feature changes; the script does not
infer behavior or claim new test results. Missing highlights or stale current
documentation versions abort preparation before any tracked file is written.
`check` also verifies the README version/changelog link and current documentation
references before publication. It does not change tracked files and accepts an
uncommitted release PR worktree:

~~~bash
python3 scripts/prepare-release.py check --notes-file /tmp/werk-release-notes.md
python3 -m unittest discover -s scripts/tests -p 'test_prepare_release.py'
~~~

## Maintainer release checklist

Prepare steps 1 and 2 in the release PR and merge it. The manual workflow then
handles step 6 and creates a draft. Check out that tag on each build host before
packaging; publish the draft only after uploading and checking the artifacts.

1. Synchronize the intended product version in `Cargo.toml`, the root package
   entry in `Cargo.lock`, `COMPANION_VERSION` in
   `runtime/werk_media_companion.py`, `utils/comfyUI/pyproject.toml`, and the
   root package entries in `utils/n8n/package.json` and its lockfile. Dependency,
   protocol and schema versions follow their own compatibility rules.
2. Move the completed changes from `Unreleased` into the dated release section
   in `CHANGELOG.md`, update its comparison links, the README release highlights
   and versioned installation examples. Run the Rust, Python probe, companion,
   ComfyUI and n8n checks described in [Building from source](build.md) and the
   [n8n validation guide](https://github.com/philipbodenbach/werk1112/blob/main/utils/n8n/docs/validation.md). Validate and pack
   the ComfyUI Registry archive; keep the integrations' Beta status explicit.
3. Build/package every target on its matching host.
4. Inspect archive contents and verify every checksum.
5. Smoke-test the extracted binary on the target operating system.
6. Create the matching `v<VERSION>` release tag.
7. Upload all five archives and their five checksum files to the GitHub
   release.
8. Test each public installer against that release.

ComfyUI Registry publication is a separate manual workflow dispatch on the
default branch, after its validation and archive checks pass. The n8n package
remains private and is distributed through the documented manual
custom-directory installation; preparing the product release does not publish
it to npm. Optional oMLX requires a separate installed runtime, and its native
Apple Silicon smoke tests must be distinguished from simulated preflight and
HTTP tests.

The repository currently has no checked-in GitHub Actions workflow that builds
or publishes the Werk release archives. The steps above remain a manual or
externally orchestrated release process.

## Known packaging limitations

- no automatic multi-platform release workflow in this repository;
- no single-host aggregate package command; each profile is built on its
  matching host;
- archives include the root README but not the complete documentation tree;
- the scripts do not produce SBOM, signature or provenance attestations;
- package validation checks archive construction, not inference on every
  optional backend.

## Related documentation

- [Building Werk1112 from source](build.md)
- [Backends, routing and platform support](../backends.md)
- [Documentation home](../documentation.md)
