#!/bin/bash
# Pumps libcamera frames into a v4l2loopback device so cv2.VideoCapture (and
# any other V4L2 client) can read the IPU6 webcam on the X1 Carbon Gen 11.
#
# Usage: scripts/start_libcamera_loopback.sh [device_index] [width] [height]
# Defaults: /dev/video42, 1280x720.
#
# Prerequisites (one-time setup):
#   sudo apt install v4l2loopback-dkms v4l-utils
#   sudo tee /etc/modules-load.d/v4l2loopback.conf <<<'v4l2loopback'
#   sudo tee /etc/modprobe.d/v4l2loopback.conf <<<'options v4l2loopback devices=1 video_nr=42 card_label="MotifikaCam" exclusive_caps=1'

set -euo pipefail

DEV_INDEX="${1:-42}"
WIDTH="${2:-1280}"
HEIGHT="${3:-720}"
DEVICE="/dev/video${DEV_INDEX}"

if [[ ! -e "$DEVICE" ]]; then
    echo "loopback device $DEVICE not found — load with:" >&2
    echo "  sudo modprobe v4l2loopback devices=1 video_nr=${DEV_INDEX} card_label=MotifikaCam exclusive_caps=1" >&2
    exit 1
fi

export GST_PLUGIN_PATH="${GST_PLUGIN_PATH:-/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/x86_64-linux-gnu"

exec gst-launch-1.0 -q libcamerasrc ! \
    "video/x-raw,width=${WIDTH},height=${HEIGHT}" ! \
    videoconvert ! \
    "video/x-raw,format=YUY2" ! \
    v4l2sink device="${DEVICE}" sync=false