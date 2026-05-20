# MOTİFİKA — Raspberry Pi (64-bit) çalıştırılabilir sürüm

Bu arşiv MOTİFİKA'nın **donmuş** sürümüdür: Raspberry Pi'de **Python kurmaya gerek
yoktur**. 64-bit Raspberry Pi OS (aarch64 / Pi 4B veya Pi 5) için derlenmiştir.
Ekran düzeni varsayılan **720p yatay** (1280×720): kamera solda, kontrol paneli sağda.
Dikey (720×1280) için: `./run.sh --portrait`.

## 1. Kurulum

```bash
tar -xzf motifika-rpi-arm64.tar.gz
cd motifika-rpi-arm64
```

## 2. Sistem bağımlılıkları (tek seferlik)

OpenCV'nin pencere/görüntüleme (Qt) ve görüntü kütüphaneleri için birkaç sistem
paketi gerekir. Raspberry Pi OS **Desktop**'ta çoğu zaten kuruludur:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libxcb1 libxcb-xinerama0 libxkbcommon-x11-0 fonts-dejavu-core
```

> Pi OS **Lite** (masaüstü yok) kullanıyorsanız bir X/Wayland oturumu da
> gerekir; bu uygulama bir ekrana pencere açar.

## 3. USB kamera

Uygulama varsayılan olarak `/dev/video0` USB kamerayı kullanır. Takılı kameraları
görmek için: `v4l2-ctl --list-devices` (gerekirse `sudo apt install v4l-utils`).
Farklı bir indeks için: `./run.sh --camera 1`.

## 4. Çalıştırma

```bash
./run.sh
```

İlk açılışta **kalibrasyon** istenir: kamera görüntüsünde tezgâhın 4 köşesine
sırayla **SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT** tıklayın. Kalibrasyon
`assets/calibration.json` dosyasına yazılır; satır/sütun/çözünürlük aynı kaldıkça
tekrar sorulmaz.

Farklı motif/boyut:

```bash
./run.sh --motif hayat_agaci --rows 40 --cols 80
./run.sh --recalibrate          # köşeleri yeniden seç
```

### Klavye

| Tuş | İşlev |
|-----|-------|
| ↑ / ↓ | aktif sırayı yukarı/aşağı kaydır |
| `r` | yeniden kalibrasyon |
| `d` | dokuma yönü değiştir (bottom_up ↔ top_down) |
| `c` | renk kontrolü aç/kapat |
| `q` / ESC | çıkış |

## 5. Ekran yönü

Varsayılan **yatay** modda görüntü 1280×720 üretilir ve 720p ekranı bire bir
doldurur. **Dikey** mod (`./run.sh --portrait`) 720×1280 üretir; bu durumda Pi
ekranını 90° döndürün (Raspberry Pi OS: *Screen Configuration* → *Orientation*,
ya da `/boot/firmware/cmdline.txt` / `wlr-randr` ile).

## 6. Sorun giderme

- **`libGL.so.1: cannot open shared object file`** → adım 2'deki `apt install`'ı çalıştırın.
- **`qt.qpa.plugin: could not load the Qt platform plugin "xcb"`** → adım 2'deki
  `libxcb*` / `libxkbcommon-x11-0` paketlerini kurun. Wayland oturumundaysanız
  `run.sh` zaten `QT_QPA_PLATFORM=xcb` ayarlar (XWayland gerekir).
- **`GLIBC_2.xx not found`** → bu ikili Pi OS **Bookworm** (Debian 12, glibc 2.36)
  veya daha yeni için derlenmiştir. Daha eski bir Pi OS (ör. Bullseye)
  kullanıyorsanız işletim sistemini güncelleyin.
- **Türkçe karakterler bozuk** → arşivdeki `fonts/` klasörünün ikilinin yanında
  durduğundan emin olun (panel fontu oradan yüklenir).
- **Kamera açılmıyor** → `--camera` indeksini deneyin (0, 1, …) ve kullanıcının
  `video` grubunda olduğundan emin olun: `sudo usermod -aG video $USER` (sonra çıkış/giriş).
