#!/bin/sh
set -eu

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

find_model_store() {
    if [ -n "${WERK_HOME:-}" ] && [ -d "$WERK_HOME" ]; then
        printf '%s\n' "$WERK_HOME"
        return 0
    fi

    if [ -n "${XDG_DATA_HOME:-}" ] && [ -d "$XDG_DATA_HOME/werk1112" ]; then
        printf '%s\n' "$XDG_DATA_HOME/werk1112"
        return 0
    fi

    if [ -n "${HOME:-}" ] && [ -d "$HOME/.local/share/werk1112" ]; then
        printf '%s\n' "$HOME/.local/share/werk1112"
        return 0
    fi

    return 1
}

find_api_keys_file() {
    if [ -n "${WERK_API_KEYS:-}" ] && [ -f "$WERK_API_KEYS" ]; then
        printf '%s\n' "$WERK_API_KEYS"
        return 0
    fi

    if [ -n "${XDG_CONFIG_HOME:-}" ] && [ -f "$XDG_CONFIG_HOME/werk1112/api-keys.toml" ]; then
        printf '%s\n' "$XDG_CONFIG_HOME/werk1112/api-keys.toml"
        return 0
    fi

    if [ -n "${HOME:-}" ] && [ -f "$HOME/.config/werk1112/api-keys.toml" ]; then
        printf '%s\n' "$HOME/.config/werk1112/api-keys.toml"
        return 0
    fi

    return 1
}

if [ "${WERK_INSTALL_DIR+x}" = "x" ]; then
    install_dir=$WERK_INSTALL_DIR
else
    [ -n "${HOME:-}" ] || die "HOME is not set; set WERK_INSTALL_DIR to choose an install directory"
    install_dir="$HOME/.local/bin"
fi

binary_path="$install_dir/werk"
model_store_kept=0
api_keys_kept=0

if [ -e "$binary_path" ]; then
    rm -f "$binary_path"
    printf 'Removed %s\n' "$binary_path"
else
    printf 'Werk1112 is not installed.\n'
fi

if model_store=$(find_model_store); then
    printf '\nWerk1112 model store detected:\n\n'
    printf '%s\n\n' "$model_store"
    printf 'This directory may contain downloaded models.\n\n'
    printf 'Remove it?\n\n'
    printf '[y/N] '

    if read answer; then
        case "$answer" in
            y|Y|yes|YES)
                rm -rf "$model_store"
                ;;
            *)
                model_store_kept=1
                ;;
        esac
    else
        model_store_kept=1
        printf '\n'
    fi
fi

if api_keys_file=$(find_api_keys_file); then
    printf '\nWerk1112 API key file detected:\n\n'
    printf '%s\n\n' "$api_keys_file"
    printf 'This file can grant access to werk serve.\n\n'
    printf 'Remove it?\n\n'
    printf '[y/N] '

    if read answer; then
        case "$answer" in
            y|Y|yes|YES)
                rm -f "$api_keys_file"
                ;;
            *)
                api_keys_kept=1
                ;;
        esac
    else
        api_keys_kept=1
        printf '\n'
    fi
fi

printf '\nWerk1112 successfully removed.\n'

if [ "$model_store_kept" -eq 1 ]; then
    printf 'Model store kept.\n'
fi

if [ "$api_keys_kept" -eq 1 ]; then
    printf 'API keys kept.\n'
fi
