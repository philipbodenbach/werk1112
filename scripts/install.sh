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

    case "$os:$arch" in
        Linux:x86_64)
            WERK_PLATFORM="linux-x86_64"
            ;;
        Darwin:arm64|Darwin:aarch64)
            WERK_PLATFORM="macos-aarch64"
            ;;
        *)
            die "unsupported OS/architecture: $os $arch"
            ;;
    esac
}

detect_downloader
detect_platform

WERK_REPO=${WERK_REPO:-phildenbo/werk1112}
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

artifact_name="werk1112-v${WERK_VERSION_NUMBER}-${WERK_PLATFORM}.tar.gz"
download_url="https://github.com/${WERK_REPO}/releases/download/${WERK_TAG}/${artifact_name}"

tmp_root=${TMPDIR:-/tmp}
tmp_dir="$tmp_root/werk1112-install-$$"
archive_path="$tmp_dir/$artifact_name"

mkdir "$tmp_dir" || die "could not create temporary directory: $tmp_dir"
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

printf 'Downloading %s\n' "$download_url"
download_to_file "$download_url" "$archive_path"

tar -xzf "$archive_path" -C "$tmp_dir"
[ -f "$tmp_dir/werk" ] || die "downloaded artifact did not contain werk"

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
