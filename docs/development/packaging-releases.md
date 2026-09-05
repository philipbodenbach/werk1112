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
- llama.cpp, vLLM, ONNX Runtime, MLX or managed Python environments;
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

For package version `1.5.1`, the generated tree is:

~~~text
releases/
├── werk1112-v1.5.1-linux-x86_64.tar.gz
├── werk1112-v1.5.1-linux-x86_64.tar.gz.sha256
├── werk1112-v1.5.1-linux-x86_64-amd-strix-halo.tar.gz
├── werk1112-v1.5.1-linux-x86_64-amd-strix-halo.tar.gz.sha256
├── werk1112-v1.5.1-linux-aarch64-dgx-spark.tar.gz
├── werk1112-v1.5.1-linux-aarch64-dgx-spark.tar.gz.sha256
├── werk1112-v1.5.1-windows-x86_64.zip
├── werk1112-v1.5.1-windows-x86_64.zip.sha256
├── werk1112-v1.5.1-macos-aarch64.tar.gz
└── werk1112-v1.5.1-macos-aarch64.tar.gz.sha256
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

## Maintainer release checklist

1. Set the intended package version in `Cargo.toml` and ensure the root package
   entry in `Cargo.lock` agrees.
2. Run the Rust, companion and relevant integration tests.
3. Build/package every target on its matching host.
4. Inspect archive contents and verify every checksum.
5. Smoke-test the extracted binary on the target operating system.
6. Create the matching `v<VERSION>` release tag.
7. Upload all five archives and their five checksum files to the GitHub
   release.
8. Test each public installer against that release.

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
