#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

cd "$REPO_ROOT"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat >&2 <<'USAGE'
Usage: ./scripts/package-release.sh <linux|linux-aarch64|windows|macos|all>

Builds release artifacts into releases/.
Artifacts are universal runtime-router binaries, one per supported OS/architecture,
with the platform accelerator path compiled in.
USAGE
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

is_dgx_spark_signal() {
    local signal="$1"
    local normalized

    normalized="$(printf '%s\n' "$signal" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' ' ')"
    case " $normalized " in
        *" NVIDIA DGX SPARK "*|*" GB10 "*) return 0 ;;
        *) return 1 ;;
    esac
}

is_dgx_spark_host() {
    local os
    local arch
    local signal

    os="$(uname -s)"
    arch="$(uname -m)"
    [[ "$os" == "Linux" ]] || return 1
    [[ "$arch" == "aarch64" || "$arch" == "arm64" ]] || return 1

    if [[ -r /proc/device-tree/model ]]; then
        signal="$(tr '\000' ' ' </proc/device-tree/model)"
        is_dgx_spark_signal "$signal" && return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if signal="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
            is_dgx_spark_signal "$signal" && return 0
        fi
    fi
    return 1
}

package_version() {
    awk -F '"' '
        /^\[package\]/ { in_package = 1; next }
        /^\[/ && in_package { exit }
        in_package && /^[[:space:]]*version[[:space:]]*=/ { print $2; exit }
    ' Cargo.toml
}

write_checksum() {
    local artifact="$1"
    local artifact_dir
    local artifact_name

    artifact_dir="$(dirname -- "$artifact")"
    artifact_name="$(basename -- "$artifact")"

    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$artifact_dir" && sha256sum "$artifact_name" > "$artifact_name.sha256")
    elif command -v shasum >/dev/null 2>&1; then
        (cd "$artifact_dir" && shasum -a 256 "$artifact_name" > "$artifact_name.sha256")
    else
        die "required checksum command not found: sha256sum or shasum"
    fi
}

package_target() {
    local platform="$1"
    local cargo_alias=""
    local binary_path=""
    local binary_name="werk"
    local artifact_name=""
    local staging_dir=""
    local artifact=""

    case "$platform" in
        linux)
            cargo_alias="build-linux"
            binary_path="target/x86_64-unknown-linux-gnu/release/werk"
            artifact_name="werk1112-v${VERSION}-linux-x86_64.tar.gz"
            ;;
        linux-aarch64)
            is_dgx_spark_host || die "linux-aarch64 release packaging must run natively on NVIDIA DGX Spark/GB10"
            cargo_alias="build-linux-aarch64"
            binary_path="target/aarch64-unknown-linux-gnu/release/werk"
            artifact_name="werk1112-v${VERSION}-linux-aarch64-dgx-spark.tar.gz"
            ;;
        windows)
            cargo_alias="build-windows"
            binary_path="target/x86_64-pc-windows-msvc/release/werk.exe"
            binary_name="werk.exe"
            artifact_name="werk1112-v${VERSION}-windows-x86_64.zip"
            ;;
        macos)
            cargo_alias="build-macos-apple-silicon"
            binary_path="target/aarch64-apple-darwin/release/werk"
            artifact_name="werk1112-v${VERSION}-macos-aarch64.tar.gz"
            ;;
        *)
            die "unsupported package target: $platform"
            ;;
    esac

    staging_dir="$REPO_ROOT/target/package/$platform"
    artifact="$REPO_ROOT/releases/$artifact_name"

    printf '\n==> Building %s release artifact\n' "$platform"
    printf '    Note: release artifacts are universal runtime-router binaries with platform accelerator support compiled in.\n'
    printf '    If cross-compilation is unavailable for your environment, build this artifact on the matching target OS.\n'
    printf '    Running: cargo %s\n' "$cargo_alias"

    if ! cargo "$cargo_alias"; then
        die "cargo $cargo_alias failed for $platform. Build this artifact on the matching target OS/toolchain if cross-compilation is unavailable."
    fi

    [ -f "$binary_path" ] || die "expected build output not found: $binary_path"

    rm -rf "$staging_dir"
    mkdir -p "$staging_dir" "$REPO_ROOT/releases"

    cp "$binary_path" "$staging_dir/$binary_name"
    cp "$REPO_ROOT/README.md" "$staging_dir/README.md"
    cp "$REPO_ROOT/LICENSE" "$staging_dir/LICENSE"
    rm -f "$artifact" "$artifact.sha256"

    case "$platform" in
        windows)
            require_command zip
            (cd "$staging_dir" && zip -q "$artifact" "$binary_name" README.md LICENSE)
            ;;
        linux|linux-aarch64|macos)
            require_command tar
            tar -czf "$artifact" -C "$staging_dir" "$binary_name" README.md LICENSE
            ;;
    esac

    write_checksum "$artifact"

    printf '    Wrote %s\n' "${artifact#$REPO_ROOT/}"
    printf '    Wrote %s.sha256\n' "${artifact#$REPO_ROOT/}"
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi

VERSION="$(package_version)"
[ -n "$VERSION" ] || die "could not read package version from Cargo.toml"

mkdir -p "$REPO_ROOT/releases" "$REPO_ROOT/target/package"

case "$1" in
    linux|linux-aarch64|windows|macos)
        package_target "$1"
        ;;
    all)
        package_target linux
        package_target linux-aarch64
        package_target windows
        package_target macos
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
