#!/bin/bash
# MOTİFİKA — Raspberry Pi (arm64) TEK KOMUT kurucu.
#
# Pi'de tek satır yeter (Python GEREKMEZ):
#     wget -qO- https://raw.githubusercontent.com/buraksirali/motifika/main/packaging/install.sh | bash
# veya indirip çalıştır:
#     wget https://raw.githubusercontent.com/buraksirali/motifika/main/packaging/install.sh
#     bash install.sh
#
# Yaptıkları: son GitHub Release'ini indirir → ~/motifika içine açar →
# masaüstü/menü kısayolunu oluşturur. Hedef klasörü değiştirmek için:
#     MOTIFIKA_DIR=/opt/motifika bash install.sh
set -euo pipefail

REPO="buraksirali/motifika"
PKG="motifika-rpi-arm64"
INSTALL_DIR="${MOTIFIKA_DIR:-$HOME/motifika}"
URL="https://github.com/$REPO/releases/latest/download/$PKG.tar.gz"

echo "[motifika] son sürüm indiriliyor:"
echo "           $URL"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if ! wget -O "$TMP/$PKG.tar.gz" "$URL"; then
    echo "[motifika] indirme başarısız. Henüz yayınlanmış bir release var mı?" >&2
    echo "           https://github.com/$REPO/releases" >&2
    exit 1
fi

echo "[motifika] $INSTALL_DIR içine açılıyor..."
mkdir -p "$INSTALL_DIR"
# Arşivin tepe klasörü ($PKG) atlanır → içerik doğrudan INSTALL_DIR'e gelir.
tar -xzf "$TMP/$PKG.tar.gz" -C "$INSTALL_DIR" --strip-components=1

cd "$INSTALL_DIR"
chmod +x run.sh motifika create_shortcut.sh 2>/dev/null || true

# Masaüstü/menü kısayolu (release ile gelen script).
if [[ -x ./create_shortcut.sh ]]; then
    ./create_shortcut.sh || echo "[motifika] kısayol oluşturulamadı (zararsız); elle: ./run.sh"
fi

cat <<EOF

[motifika] Kuruldu: $INSTALL_DIR
[motifika] Çalıştır: masaüstü simgesi / uygulama menüsü
           ya da terminalden:  cd "$INSTALL_DIR" && ./run.sh

İlk açılışta kalibrasyon istenir (4 köşeye tıkla). Sistem bağımlılıkları için
$INSTALL_DIR/README.md adım 2'ye bakın (Pi OS Desktop'ta çoğu zaten kuruludur).
EOF
