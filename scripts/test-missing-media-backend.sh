#!/usr/bin/env bash
set -euo pipefail

# Reproducible negative smoke test for Werk's managed-backend recommendation.
# It uses an isolated model home and a tiny metadata-only Qwen VoiceDesign
# fixture, so the user's real models and installed qwen-tts backend are never
# read, changed, or removed.

werk_bin="${WERK_BIN:-werk}"
if ! command -v "$werk_bin" >/dev/null 2>&1; then
    echo "error: Werk binary not found: $werk_bin" >&2
    exit 1
fi

test_root="$(mktemp -d "${TMPDIR:-/tmp}/werk-missing-backend.XXXXXX")"
cleanup() {
    rm -rf -- "$test_root"
}
trap cleanup EXIT

model_source="$test_root/qwen-voice-design-fixture"
model_home="$test_root/store"
model_id="qwen-tts-missing-backend-smoke"
mkdir -p "$model_source"

printf '%s\n' \
    '{"model_type":"qwen3_tts","tts_model_type":"voice_design","architectures":["Qwen3TTSForConditionalGeneration"]}' \
    >"$model_source/config.json"
printf 'negative-test-placeholder\n' >"$model_source/model.safetensors"

# Runtime overrides could make the isolated test see a real external companion
# or Qwen environment. Clear them only for this script; the caller's
# environment remains unchanged.
unset WERK_MEDIA_COMPANION
unset WERK_MEDIA_COMPANION_SCRIPT
unset WERK_MEDIA_PYTHON
unset WERK_QWEN_TTS_PYTHON

"$werk_bin" --model-home "$model_home" import "$model_source" --name "$model_id" >/dev/null

echo "== Doctor output: expected installable backend =="
doctor_output="$(
    "$werk_bin" --model-home "$model_home" --no-auto-install-backends \
        doctor --model "$model_id" --task text-to-speech
)"
printf '%s\n' "$doctor_output"

if ! grep -Fq 'Task readiness: installable' <<<"$doctor_output"; then
    echo "error: doctor did not report Task readiness: installable" >&2
    exit 1
fi
if ! grep -Fq 'Recommendation: werk backend install qwen-tts' <<<"$doctor_output"; then
    echo "error: doctor did not report the managed qwen-tts install command" >&2
    exit 1
fi

echo
echo "== Generation output: expected preflight failure and recommendation =="
set +e
generation_output="$(
    "$werk_bin" --model-home "$model_home" --no-auto-install-backends \
        audio generate speech "$model_id" \
        --text 'This request must stop during backend preflight.' \
        --backend auto \
        --fallback-policy none \
        --format wav \
        --output "$test_root/must-not-exist.wav" \
        --debug 2>&1
)"
generation_status=$?
set -e
printf '%s\n' "$generation_output"

if ((generation_status == 0)); then
    echo "error: generation unexpectedly succeeded without qwen-tts" >&2
    exit 1
fi
if ! grep -Fq 'Recommendation: run `werk backend install qwen-tts`' <<<"$generation_output"; then
    echo "error: failed generation did not include the qwen-tts recommendation" >&2
    exit 1
fi
if [[ -e "$test_root/must-not-exist.wav" ]]; then
    echo "error: preflight failure unexpectedly created an output file" >&2
    exit 1
fi

echo
echo "PASS: missing managed backend was detected before inference and no output was created."
