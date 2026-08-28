#!/bin/sh
set -eu

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

detect_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DOWNLOADER="curl"
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER="wget"
    else
        die "curl or wget is required to download Werk1112"
    fi
}

detect_checksum_tool() {
    if command -v sha256sum >/dev/null 2>&1; then
        CHECKSUM_TOOL="sha256sum"
    elif command -v shasum >/dev/null 2>&1; then
        CHECKSUM_TOOL="shasum"
    else
        die "sha256sum or shasum is required to verify the Werk1112 release"
    fi
}

is_dgx_spark_signal() {
    signal=$1
    normalized=$(printf '%s\n' "$signal" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' ' ')

    case " $normalized " in
        *" NVIDIA DGX SPARK "*|*" GB10 "*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_dgx_spark_host() {
    if [ -r /proc/device-tree/model ]; then
        device_model=$(tr '\000' ' ' </proc/device-tree/model)
        if is_dgx_spark_signal "$device_model"; then
            return 0
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        if gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
            if is_dgx_spark_signal "$gpu_names"; then
                return 0
            fi
        fi
    fi

    return 1
}

is_strix_halo_signal() {
    signal=$1
    normalized=$(printf '%s\n' "$signal" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' ' ')

    case " $normalized " in
        *" AMD RYZEN AI MAX "*|*" STRIX HALO "*|*" RADEON 8060S "*|*" RADEON 8050S "*|*" RADEON 8040S "*|*" GFX1151 "*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_strix_halo_host() {
    [ "$(uname -s)" = "Linux" ] || return 1
    [ "$(uname -m)" = "x86_64" ] || return 1

    if [ -r /proc/cpuinfo ]; then
        cpu_info=$(sed -n 's/^[Mm]odel name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo)
        if is_strix_halo_signal "$cpu_info"; then
            return 0
        fi
    fi

    for dmi_path in /sys/class/dmi/id/product_name /sys/class/dmi/id/board_name; do
        if [ -r "$dmi_path" ]; then
            dmi_info=$(sed -n '1p' "$dmi_path")
            if is_strix_halo_signal "$dmi_info"; then
                return 0
            fi
        fi
    done

    if command -v lscpu >/dev/null 2>&1; then
        if cpu_info=$(lscpu 2>/dev/null); then
            if is_strix_halo_signal "$cpu_info"; then
                return 0
            fi
        fi
    fi

    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        if gpu_agents=$(rocm_agent_enumerator 2>/dev/null); then
            if is_strix_halo_signal "$gpu_agents"; then
                return 0
            fi
        fi
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        if gpu_info=$(rocminfo 2>/dev/null); then
            if is_strix_halo_signal "$gpu_info"; then
                return 0
            fi
        fi
    fi

    return 1
}

download_to_file() {
    url=$1
    output=$2

    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fL "$url" -o "$output"
    else
        wget -O "$output" "$url"
    fi
}

download_to_stdout() {
    url=$1

    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fsSL "$url"
    else
        wget -qO- "$url"
    fi
}

verify_checksum() {
    directory=$1
    checksum_name=$2

    if [ "$CHECKSUM_TOOL" = "sha256sum" ]; then
        (cd "$directory" && sha256sum -c "$checksum_name")
    else
        (cd "$directory" && shasum -a 256 -c "$checksum_name")
    fi
}

validate_archive_listing() {
    archive=$1

    if ! archive_entries=$(tar -tzf "$archive"); then
        die "could not read downloaded release archive"
    fi
    sorted_entries=$(printf '%s\n' "$archive_entries" | LC_ALL=C sort)
    expected_entries=$(printf '%s\n' LICENSE README.md werk)
    [ "$sorted_entries" = "$expected_entries" ] || die "release archive contains unexpected entries"
}

validate_extracted_archive() {
    directory=$1

    for name in werk README.md LICENSE; do
        [ -f "$directory/$name" ] || die "release archive did not contain regular file: $name"
        [ ! -L "$directory/$name" ] || die "release archive contained a symbolic link: $name"
    done
}

normalize_version() {
    input=$1

    case "$input" in
        v*)
            WERK_TAG=$input
            WERK_VERSION_NUMBER=${input#v}
            ;;
        *)
            WERK_TAG="v$input"
            WERK_VERSION_NUMBER=$input
            ;;
    esac
}

detect_platform() {
    os=$(uname -s)
    arch=$(uname -m)
    WERK_FALLBACK_PLATFORM=""

    case "$os:$arch" in
        Linux:x86_64)
            if is_strix_halo_host; then
                WERK_PLATFORM="linux-x86_64-amd-strix-halo"
                WERK_FALLBACK_PLATFORM="linux-x86_64"
            else
                WERK_PLATFORM="linux-x86_64"
            fi
            ;;
        Linux:arm64|Linux:aarch64)
            is_dgx_spark_host || die "unsupported Linux aarch64 host: the prebuilt arm64 release is limited to NVIDIA DGX Spark/GB10; build Werk from source on other ARM64 systems"
            WERK_PLATFORM="linux-aarch64-dgx-spark"
            ;;
        Darwin:arm64|Darwin:aarch64)
            WERK_PLATFORM="macos-aarch64"
            ;;
        *)
            die "unsupported OS/architecture: $os $arch"
            ;;
    esac
}

configure_artifact() {
    platform=$1
    artifact_name="werk1112-v${WERK_VERSION_NUMBER}-${platform}.tar.gz"
    download_url="https://github.com/${WERK_REPO}/releases/download/${WERK_TAG}/${artifact_name}"
    checksum_name="$artifact_name.sha256"
    checksum_url="$download_url.sha256"
    archive_path="$tmp_dir/$artifact_name"
    checksum_path="$tmp_dir/$checksum_name"
}

download_artifact_and_checksum() {
    printf 'Downloading %s\n' "$download_url"
    if ! download_to_file "$download_url" "$archive_path"; then
        return 1
    fi
    printf 'Downloading %s\n' "$checksum_url"
    if ! download_to_file "$checksum_url" "$checksum_path"; then
        return 1
    fi
    return 0
}

detect_downloader
detect_checksum_tool
detect_platform

WERK_REPO=${WERK_REPO:-philipbodenbach/werk1112}
WERK_VERSION_INPUT=${WERK_VERSION:-latest}

if [ "${WERK_INSTALL_DIR+x}" = "x" ]; then
    install_dir=$WERK_INSTALL_DIR
else
    [ -n "${HOME:-}" ] || die "HOME is not set; set WERK_INSTALL_DIR to choose an install directory"
    install_dir="$HOME/.local/bin"
fi

if [ "$WERK_VERSION_INPUT" = "latest" ]; then
    latest_json=$(download_to_stdout "https://api.github.com/repos/$WERK_REPO/releases/latest")
    latest_tag=$(printf '%s\n' "$latest_json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | sed -n '1p')
    [ -n "$latest_tag" ] || die "could not resolve latest release for $WERK_REPO"
    normalize_version "$latest_tag"
else
    normalize_version "$WERK_VERSION_INPUT"
fi

tmp_root=${TMPDIR:-/tmp}
tmp_dir="$tmp_root/werk1112-install-$$"

mkdir "$tmp_dir" || die "could not create temporary directory: $tmp_dir"
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

configure_artifact "$WERK_PLATFORM"
if ! download_artifact_and_checksum; then
    if [ -z "$WERK_FALLBACK_PLATFORM" ]; then
        die "could not download release artifact and checksum for $WERK_PLATFORM"
    fi
    printf 'Warning: release %s has no usable %s archive; falling back to %s.\n' \
        "$WERK_TAG" "$WERK_PLATFORM" "$WERK_FALLBACK_PLATFORM" >&2
    configure_artifact "$WERK_FALLBACK_PLATFORM"
    download_artifact_and_checksum || die "could not download fallback release artifact and checksum for $WERK_FALLBACK_PLATFORM"
fi

printf 'Verifying %s\n' "$artifact_name"
verify_checksum "$tmp_dir" "$checksum_name" || die "checksum verification failed for $artifact_name"
validate_archive_listing "$archive_path"

tar -xzf "$archive_path" -C "$tmp_dir"
validate_extracted_archive "$tmp_dir"

mkdir -p "$install_dir"
cp "$tmp_dir/werk" "$install_dir/werk"
chmod +x "$install_dir/werk"

printf 'Installed %s\n' "$install_dir/werk"

case ":${PATH:-}:" in
    *":$install_dir:"*) ;;
    *) printf 'Warning: %s is not on PATH. Add it to PATH to run werk from any directory.\n' "$install_dir" >&2 ;;
esac

printf '\nWerk1112 installed successfully.\n\n'
printf 'Run:\n'
printf '  werk --help\n'
