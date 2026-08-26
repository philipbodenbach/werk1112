# Building Werk1112 from source

This page documents the current Cargo build surface. It is intended for
contributors, maintainers and users who need a custom binary. Normal end users
should prefer the installers in the [project README](../../README.md#install).

The authoritative build definitions are:

- [`Cargo.toml`](../../Cargo.toml) for features and release profiles;
- [`.cargo/config.toml`](../../.cargo/config.toml) for Cargo aliases and default
  build environment values;
- [`rust-toolchain.toml`](../../rust-toolchain.toml) for the Rust channel,
  components and target triples;
- [`scripts/build-windows.ps1`](../../scripts/build-windows.ps1) for the guarded
  native Windows release build.

See [Packaging and releases](packaging-releases.md) after producing a target
binary. Runtime installation and operating-system support are separate topics;
see [Backends, routing and platform support](../backends.md).

## Build features and runtime backends are different

Cargo features decide which in-process Rust acceleration code is compiled into
the Werk binary. They do not install CUDA, ROCm, llama.cpp, vLLM, ONNX Runtime,
MLX, Python packages, media codecs or model weights.

External and managed runtimes are discovered when Werk starts. For example, the
normal GGUF hot path uses a separate persistent `llama-server`; compiling the
Werk `cuda` feature does not compile that server. Provision it independently
with commands such as:

~~~bash
werk backend install llama-cuda
werk backend doctor --debug
~~~

Likewise, selecting `--backend cuda`, `--backend vllm` or another runtime at
execution time is not the same as choosing a Cargo feature at build time.

## Rust toolchain

The repository pins the stable Rust channel and requests `rustfmt`, `clippy`
and these release targets:

~~~text
x86_64-unknown-linux-gnu
x86_64-pc-windows-msvc
aarch64-unknown-linux-gnu
aarch64-apple-darwin
~~~

Use the explicit `+stable` selector in source-install instructions. This also
avoids Cargo's warning about an implicitly selected repository toolchain during
`cargo install`.

Basic contributor checks are:

~~~bash
cargo +stable fmt --all --check
cargo +stable check --locked --all-targets
cargo +stable test --locked
~~~

Python companion and ComfyUI tests are independent of the Rust build:

~~~bash
python -m unittest runtime.test_werk_media_companion
python -m pytest utils/comfyUI/tests
~~~

## Backend-neutral development build

The current Cargo `default` feature set is empty. A normal source build
therefore does not require a CUDA or Metal toolchain:

~~~bash
cargo +stable check --locked
cargo +stable build --locked
~~~

The debug binary is written to `target/debug/werk`. Build a locked release
binary with the checked-in CPU alias:

~~~bash
cargo +stable build-cpu
~~~

`build-cpu` expands to:

~~~text
build --release --locked --no-default-features
~~~

Its binary is `target/release/werk`. Despite the alias name, external runtimes
can still be discovered by that binary; the name describes the compiled
in-process feature set.

Install the backend-neutral binary on the current Cargo `PATH`:

~~~bash
cargo +stable install --path . --locked --force
~~~

`cargo build` only creates a file below `target/`. It never replaces an older
`werk` already found on `PATH`. Run the local file explicitly or use
`cargo install` when replacement is intended.

## Target release aliases

Release tooling uses one alias for each configured operating-system and
architecture pair:

| Alias | Host normally used | Feature bundle | Output |
| --- | --- | --- | --- |
| `cargo +stable build-linux` | Native Linux or WSL | `release-linux` | `target/x86_64-unknown-linux-gnu/release/werk` |
| `cargo +stable build-linux-aarch64` | Native DGX Spark/GB10 | `release-linux-aarch64` | `target/aarch64-unknown-linux-gnu/release/werk` |
| `cargo +stable build-windows` | Native Windows Developer PowerShell | `release-windows` | `target/x86_64-pc-windows-msvc/release/werk.exe` |
| `cargo +stable build-macos-apple-silicon` | Apple Silicon macOS | `release-macos-apple-silicon` | `target/aarch64-apple-darwin/release/werk` |

All four aliases use `--release --locked --no-default-features` and an
explicit target triple. Their bundle definitions in `Cargo.toml` are:

| Bundle | Current expansion | Compiled accelerator path |
| --- | --- | --- |
| `release-linux` | `cuda` | Candle CPU and Candle CUDA |
| `release-linux-aarch64` | `cuda` | Candle CPU and Candle CUDA, compiled for DGX Spark `sm_121` |
| `release-windows` | `cuda` | Candle CPU and Candle CUDA |
| `release-macos-apple-silicon` | `metal` | Candle CPU and Candle Metal |

The Windows alias additionally applies these command-local environment values:

~~~text
CL=/Zc:preprocessor
NVCC_PREPEND_FLAGS=-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
~~~

They are the CUDA 13.3/MSVC preprocessor compatibility settings and do not
apply to Linux or macOS aliases.

The corresponding raw Linux and macOS commands are:

~~~bash
cargo +stable build --release --locked --no-default-features \
  --target x86_64-unknown-linux-gnu --features release-linux

CUDA_COMPUTE_CAP=121 cargo +stable build \
  --release --locked --no-default-features \
  --target aarch64-unknown-linux-gnu --features release-linux-aarch64

cargo +stable build --release --locked --no-default-features \
  --target aarch64-apple-darwin --features release-macos-apple-silicon
~~~

Prefer the checked-in alias or Windows build script for Windows so the two
MSVC compatibility values cannot be omitted accidentally.

Having a Rust target installed is not sufficient for arbitrary
cross-compilation. The target linker, SDK and accelerator toolchain must also
be available. Build each release on its matching platform unless a complete
cross-toolchain has been configured.

## Cargo feature reference

The feature graph currently exposed by `Cargo.toml` is:

| Feature | Purpose |
| --- | --- |
| `default` | Empty; backend-neutral source build. |
| `recommended` | Convenience preset expanding to `cuda`; it is not a default feature. |
| `cuda` | Alias for `candle-cuda`. |
| `candle-cuda` | CUDA support in Candle core, neural-network and transformer crates. |
| `cudnn` | `candle-cuda` plus Candle cuDNN support. |
| `metal` | Candle Metal support for macOS. |
| `mkl` | Candle MKL support. |
| `llama-cpp` | Low-level optional `llama_cpp` dependency. |
| `llama-fast` | Low-level optional `llama_cpp_sys` dependency. |
| `llama-legacy-cuda` | Old in-process llama.cpp FFI path with CUDA. |
| `llama-legacy-vulkan` | Old in-process llama.cpp FFI path with Vulkan. |
| `vulkan` | Compatibility alias for `llama-legacy-vulkan`. |
| `cuda-mmq` | Compatibility alias for `llama-legacy-cuda`; it does not force the old `cuda_mmq` dependency feature. |
| `burn-experimental` | Enables the experimental Burn command/routing surface. |
| `burn-runtime` | Burn runtime and model-store dependencies. |
| `burn-cpu` | Burn runtime with the CPU/flex backend. |
| `burn-cuda` | Burn runtime with CUDA/fusion and the `cudarc` CUDA 12.8 selector. |
| `release-linux` | Linux x86_64 release feature bundle; currently `cuda`. |
| `release-linux-aarch64` | DGX Spark/Linux aarch64 release feature bundle; currently `cuda`. |
| `release-windows` | Windows release feature bundle; currently `cuda`. |
| `release-macos-apple-silicon` | Apple Silicon release feature bundle; currently `metal`. |

The persistent llama.cpp server route is preferred over the legacy in-process
FFI features. Build those legacy features only for development or regression
testing.

## Linux x86_64 release build

`release-linux` currently includes Candle CUDA, so producing the configured
Linux artifact requires a working CUDA build environment even though the final
Werk router can later use CPU or external runtimes.

The repository supplies these non-forced defaults in `.cargo/config.toml`:

~~~text
CUDA_COMPUTE_CAP=86
CC_x86_64_unknown_linux_gnu=gcc-10
CXX_x86_64_unknown_linux_gnu=g++-10
~~~

An existing shell value wins. Compute capability `86` targets Ampere cards
such as an RTX 3090; set the value appropriate for the binary you intend to
build. Install GCC/G++ 10 or override the two target-specific compiler values
when those executables are not available.

A typical CUDA environment looks like this; replace the example path with the
toolkit actually installed on the build host:

~~~bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_ROOT="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvcc --version
cargo +stable build-linux
~~~

Some distributions additionally need Clang and libclang for native dependency
build scripts. On Debian/Ubuntu that can be installed with:

~~~bash
sudo apt-get update
sudo apt-get install -y clang libclang-dev gcc-10 g++-10
~~~

For an explicitly installed custom CUDA build rather than the release alias:

~~~bash
cargo +stable install --path . --locked --force --features cuda
~~~

If Candle reports `fatal error: cuda_fp8.h: No such file or directory`, the
active toolkit is too old for the selected Candle CUDA code. Verify that the
intended toolkit's `bin`, `include` and library directories precede an older
distribution CUDA package.

## DGX Spark / Linux aarch64 release build

Werk's configured Linux aarch64 target is built for the NVIDIA DGX Spark GB10.
GB10 is an arm64 system with CUDA compute capability 12.1 (`sm_121`). The
checked-in alias therefore overrides the repository's x86/Ampere default and
sets `CUDA_COMPUTE_CAP=121` for this command only:

~~~bash
rustup target add aarch64-unknown-linux-gnu
nvcc --version
cargo +stable build-linux-aarch64
~~~

Produce and smoke-test the release artifact natively on a DGX Spark (or an
equivalent native Linux aarch64/GB10 release builder). The repository does not
ship an aarch64 CUDA cross-toolchain, sysroot, or containerized cross-build.
Specifying the Rust target alone on an x86_64 host is not a supported release
build.

The Spark build requires a CUDA toolkit capable of compiling `sm_121`; use
CUDA 13 or newer and ensure its `nvcc`, headers and libraries are selected.
DGX OS already provides an arm64 NVIDIA software base, but optional runtimes
such as llama.cpp, vLLM, TensorRT-LLM and model-specific containers remain
separate installations. Building Werk does not install them.

The resulting binary is:

~~~text
target/aarch64-unknown-linux-gnu/release/werk
~~~

The compute-capability value follows NVIDIA's
[CUDA GPU table](https://developer.nvidia.com/cuda-gpus) and
[DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/).

## Native Windows x64 release build

Build the Windows artifact on native Windows, not inside WSL. Use a Windows
filesystem checkout such as `C:\dev\werk1112`, not a `\\wsl$` path.

Prerequisites are:

1. Rustup with the `stable-x86_64-pc-windows-msvc` toolchain;
2. Visual Studio Build Tools with **Desktop development with C++**;
3. an x64 Developer PowerShell or x64 Native Tools shell;
4. a Windows CUDA Toolkit with `nvcc.exe` on `PATH`;
5. Git and Git LFS for repository/model workflows;
6. LLVM/libclang when required by native dependency build scripts.

The guarded build entry point is:

~~~powershell
.\scripts\build-windows.ps1
~~~

The script:

- refuses to run outside native Windows;
- adds the two CUDA/MSVC compatibility flags without overwriting existing
  process flags;
- requires `cl.exe` and `nvcc.exe`;
- verifies that `cl.exe` comes from a `Hostx64\x64` toolchain;
- runs the same target and feature selection as `cargo build-windows`.

To initialize Developer PowerShell programmatically:

~~~powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = & $vswhere -latest -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
  -property installationPath

if (-not $vsInstall) {
  throw "Visual Studio C++ build tools were not found."
}

Import-Module (Join-Path $vsInstall "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsInstall -SkipAutomaticLocation `
  -DevCmdArguments "-arch=x64 -host_arch=x64"

where.exe cl
where.exe nvcc
.\scripts\build-windows.ps1
~~~

If `rustup default stable-x86_64-pc-windows-msvc` reports that the toolchain
cannot run, or the prompt starts in `\\wsl.localhost`, the command is probably
running in WSL rather than native Windows. Move to Windows PowerShell before
building the Windows artifact.

## Apple Silicon macOS release build

The published macOS target is Apple Silicon only. Install Xcode command-line
tools and the Rust target, then run on an arm64 Mac:

~~~bash
xcode-select --install
rustup target add aarch64-apple-darwin
cargo +stable build-macos-apple-silicon
~~~

The bundle enables Candle Metal and writes
`target/aarch64-apple-darwin/release/werk`. There is currently no published
macOS x86_64 release target.

## Optional and experimental builds

Custom Candle builds:

~~~bash
cargo +stable build --release --locked --features mkl
cargo +stable build --release --locked --features candle-cuda
cargo +stable build --release --locked --features cuda,cudnn
~~~

`cudnn` requires a compatible cuDNN installation in addition to CUDA.

Experimental Burn builds:

~~~bash
cargo +stable install --path . --locked --force --features burn-cpu
cargo +stable install --path . --locked --force --features burn-cuda
cargo +stable install --path . --locked --force --no-default-features \
  --features cuda,burn-cpu,burn-cuda
~~~

Burn CUDA requires native CUDA and NCCL libraries. Install them through the
operating-system package manager or provide native `CUDA_HOME` and `NCCL_HOME`
locations. `werk backend doctor --debug` reports CUDA runtime, CUDA stub and
NCCL discovery plus the Burn CUDA smoke-test result.

Legacy in-process llama.cpp builds:

~~~bash
cargo +stable install --path . --locked --force --features llama-legacy-cuda
cargo +stable install --path . --locked --force --features llama-legacy-vulkan
~~~

For normal GGUF execution, install a persistent server backend instead:

~~~bash
werk backend install llama-cuda
~~~

## Troubleshooting

### `can't find crate for core` or `E0463`

Install the target for the active stable toolchain:

~~~bash
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-pc-windows-msvc
rustup target add aarch64-apple-darwin
~~~

Only the target matching the intended build is required. A native SDK/linker
may still be necessary after the Rust standard library is installed.

### A removed feature still appears in a linker error

Clear only release artifacts and rebuild:

~~~bash
cargo +stable clean --release
~~~

Do not delete the global Cargo registry as a routine build fix.

### `gcc-10` or `g++-10` is not found

Install those compilers or override the repository defaults for the target:

~~~bash
export CC_x86_64_unknown_linux_gnu=gcc
export CXX_x86_64_unknown_linux_gnu=g++
cargo +stable build-linux
~~~

The replacement compiler must still be compatible with the selected CUDA
toolkit.

### Windows reports missing or wrong `cl.exe`

Open an x64 Developer PowerShell or use the `VsDevShell` snippet above. The
checked-in Windows script intentionally rejects an x86-hosted compiler path.

### Windows reports missing `nvcc.exe`

Install the CUDA Toolkit, not only the NVIDIA display driver, and ensure the
toolkit's `bin` directory is on the current process `PATH`.

### Burn CUDA reports CUDA stubs or missing NCCL

Use native driver/runtime libraries. In WSL, keep `/usr/lib/wsl/lib` visible
and remove CUDA stub directories from `LD_LIBRARY_PATH`. Install NCCL or set
`NCCL_HOME` to a directory containing the native NCCL library.

## Next steps

- [Package target binaries and checksums](packaging-releases.md)
- [Install and diagnose runtime backends](../backends.md)
- [Return to the documentation home](../README.md)
