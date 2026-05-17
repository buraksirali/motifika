#!/bin/bash
# One-shot launcher: ensures the libcamera->loopback pump is running, then
# runs `python -m app.main` with whatever args you pass.
#
# Examples:
#   scripts/run.sh --motif eli_belinde --rows 30 --cols 60
#   scripts/run.sh --motif hayat_agaci --rows 40 --cols 80 --recalibrate
#
# The camera service stays running after the app exits. Stop it manually with:
#   systemctl --user stop motifika-camera.service

set -euo pipefail

SERVICE="motifika-camera.service"
DEVICE="/dev/video42"

cd "$(dirname "$0")/.."

if ! systemctl --user is-active --quiet "$SERVICE"; then
    echo "[run] starting camera pump..."
    systemctl --user start "$SERVICE"
    for _ in {1..20}; do
        sleep 0.25
        if systemctl --user is-active --quiet "$SERVICE" && [[ -e "$DEVICE" ]]; then
            break
        fi
    done
    if ! systemctl --user is-active --quiet "$SERVICE"; then
        echo "[run] camera pump failed to start. Check:" >&2
        echo "  journalctl --user -u $SERVICE -n 30 --no-pager" >&2
        exit 1
    fi
fi

# Default to --camera 42 (the loopback) if the user didn't override it.
HAS_CAMERA_FLAG=0
HAS_IMAGE_FLAG=0
HAS_FULLSCREEN_FLAG=0
for arg in "$@"; do
    [[ "$arg" == "--camera" ]] && HAS_CAMERA_FLAG=1
    [[ "$arg" == "--image" ]] && HAS_IMAGE_FLAG=1
    [[ "$arg" == "--fullscreen" ]] && HAS_FULLSCREEN_FLAG=1
done

if [[ $HAS_CAMERA_FLAG -eq 0 && $HAS_IMAGE_FLAG -eq 0 ]]; then
    set -- "$@" --camera 42
fi

# run.sh varsayılan olarak tam ekran açar; istemiyorsan app.main'i doğrudan çağır.
if [[ $HAS_FULLSCREEN_FLAG -eq 0 ]]; then
    set -- "$@" --fullscreen
fi

exec python3 -m app.main "$@"