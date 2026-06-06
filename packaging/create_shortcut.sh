#!/bin/bash
# MOTİFİKA — masaüstü + uygulama menüsü kısayolu oluşturur.
#
# Donmuş Raspberry Pi sürümüyle BİRLİKTE gelir (release.yml paketin içine kopyalar).
# Çıkardığın motifika klasöründe çalıştır:
#     ./create_shortcut.sh
#
# Kısayolu hem uygulama menüsüne (~/.local/share/applications) hem de varsa
# masaüstüne (~/Desktop) yazar. Exec mutlak yolla run.sh'i çağırır → kısayol
# klasör nereye çıkarılmışsa oraya bağlanır (taşınırsa tekrar çalıştır).
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$APP_DIR/run.sh"

if [[ ! -x "$RUN" ]]; then
    echo "HATA: run.sh bulunamadı ($RUN). Bu scripti çıkarılan motifika klasöründen çalıştır." >&2
    exit 1
fi

# İkon: önce motif görseli, sonra pikselli önizleme, en son genel bir tema ikonu.
ICON="utilities-graphics"
for cand in "$APP_DIR/eli_belinde.jpeg" \
            "$APP_DIR/assets/eli_belinde_chart.preview.png" \
            "$APP_DIR/hayat_agaci.jpeg"; do
    if [[ -f "$cand" ]]; then ICON="$cand"; break; fi
done

DESKTOP_NAME="motifika.desktop"
ENTRY="[Desktop Entry]
Version=1.0
Type=Application
Name=MOTİFİKA
Comment=AR kilim/halı dokuma rehberi
Exec=$RUN
Path=$APP_DIR
Icon=$ICON
Terminal=false
Categories=Education;Graphics;
"

# 1) Uygulama menüsü (her zaman).
APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
printf '%s' "$ENTRY" > "$APPS_DIR/$DESKTOP_NAME"
chmod +x "$APPS_DIR/$DESKTOP_NAME"
echo "Menü kısayolu: $APPS_DIR/$DESKTOP_NAME"

# 2) Masaüstü (varsa). xdg-user-dir lokalize "Masaüstü" yolunu verir; yoksa ~/Desktop.
DESK_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[[ -z "${DESK_DIR:-}" || ! -d "$DESK_DIR" ]] && DESK_DIR="$HOME/Desktop"
if [[ -d "$DESK_DIR" ]]; then
    cp "$APPS_DIR/$DESKTOP_NAME" "$DESK_DIR/$DESKTOP_NAME"
    chmod +x "$DESK_DIR/$DESKTOP_NAME"
    # Pi OS / GNOME dosya yöneticileri çift tıkta çalışması için "güvenilir" ister.
    gio set "$DESK_DIR/$DESKTOP_NAME" metadata::trusted true 2>/dev/null || true
    echo "Masaüstü kısayolu: $DESK_DIR/$DESKTOP_NAME"
else
    echo "Masaüstü klasörü yok; yalnızca uygulama menüsüne eklendi."
fi

echo "Tamam. MOTİFİKA'yı menüden veya masaüstü simgesinden başlatabilirsin."
