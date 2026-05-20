#!/bin/bash
# MOTİFİKA — Raspberry Pi başlatıcı (Python GEREKMEZ; donmuş ikili).
#
# Kullanım: arşivi çıkar, bu klasörden çalıştır:
#     ./run.sh
# Farklı motif/boyut:
#     ./run.sh --motif hayat_agaci --rows 40 --cols 80
# Yeniden kalibrasyon:
#     ./run.sh --recalibrate
set -euo pipefail

cd "$(dirname "$0")"

ARGS=("$@")
# Hiç argüman verilmediyse makul varsayılanlar (eli_belinde, 30×60).
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(--motif eli_belinde --rows 30 --cols 60)
fi

# Belirli bir bayrak verilmediyse ekle: USB kamera 0 + portrait düzen + tam ekran.
add_if_missing() {
    local flag="$1"; shift
    local a
    for a in "${ARGS[@]}"; do
        [[ "$a" == "$flag" ]] && return 0
    done
    ARGS+=("$@")
}
# --image verildiyse kamerayı zorlama (sabit görsel test modu).
HAS_IMAGE=0
for a in "${ARGS[@]}"; do [[ "$a" == "--image" ]] && HAS_IMAGE=1; done
[[ $HAS_IMAGE -eq 0 ]] && add_if_missing --camera --camera 0
# Varsayılan düzen YATAY (1280×720). Dikey istersen: ./run.sh --portrait
add_if_missing --fullscreen --fullscreen

# Wayland oturumunda OpenCV'nin Qt highgui'si xcb ister; XWayland üzerinden çalışır.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

exec ./motifika "${ARGS[@]}"
