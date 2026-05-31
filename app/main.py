"""MOTİFİKA ana döngü — tüm modülleri sırayla çağırıp birbirine bağlar.

Her karede: kameradan kare al → aktif sırayı bul → (her N karede) renk kontrolü
→ AR overlay çiz → sağ paneli çiz → yan yana yapıştır → ekrana bas → klavye.

Kullanım:
    python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0
    python -m app.main --motif hayat_agaci --rows 50 --cols 30 --image kilim.jpg

Başlangıçta tıklanabilir bir ekranda motif seçilir (kalibrasyondan önce).
Tüm kontroller hem KLAVYE hem de DOKUNMATIK BUTON ile yapılabilir (butonlar ekstra,
klavye korunur). Butonlar kameranın altındaki 2 satırlık ızgaradadır.

Klavye (= buton karşılığı):
    [yukarı/aşağı ok] sıra      [z]/[x] yakınlaş/uzaklaş
    [+]/[-]  saydamlık          [d] yön değiştir
    [c] renk kontrolü           [r] kalibrasyon
    [q]/ESC çıkış               (Motif: yalnız buton)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.calibration import (
    DEFAULT_PATH as CAL_PATH,
    collect_corners_interactive,
    load_calibration,
    orient_frame,
    save_calibration,
    setup_window,
)
from app.color_check import HSVBackend, check_active_row, last_completed_row
from app.overlay import DEFAULT_TRANSPARENCY, Chart, OverlayRenderer
from app.pattern import build_chart, save_chart
from app.progress import ProgressTracker
from app.ui import UIRenderer, _draw_texts


# Motif kataloğu: isim → kaynak görsel. Yeni motif = buraya satır + görsel ekle.
MOTIFS = {
    "eli_belinde": Path("eli_belinde.jpeg"),
    "hayat_agaci": Path("hayat_agaci.jpeg"),
}

# Motif seçici (başlangıç) ekranında gösterilecek Türkçe etiketler.
MOTIF_LABELS = {
    "eli_belinde": "Eli belinde",
    "hayat_agaci": "Hayat ağacı",
}

# Her N karede 1 renk kontrolü. Kontrol pahalı; 5 = göze anlık görünür + CPU yormaz.
COLOR_CHECK_EVERY_N = 5

# Yatay (landscape) düzen: 720p ekran = 1280 geniş × 720 yüksek. Kamera solda
# (880×720 kutuya 16:9 letterbox), panel sağda (400×720) → birleşik tam 1280×720,
# ekranı bozulmadan doldurur. Kamera zoom/çözünürlük değişse de kutu sabit kalır.
LANDSCAPE_SCREEN = (1280, 720)       # (W, H) — fiziksel ekran çözünürlüğü
LANDSCAPE_PANEL_WIDTH = 400          # sağ panel genişliği (kamera kutusu = 1280-400)

# Dikey (portrait) düzen: Raspberry Pi 720p portrait ekran = 720 geniş × 1280 yüksek.
# Üstte 16:9 kamera kutusu (720×405), altta panel kalan yüksekliği (1280-405=875) doldurur.
# compose() yatay yerine dikey birleştirir; panel tüm ekran genişliğinde (720) çizilir.
PORTRAIT_SCREEN = (720, 1280)        # (W, H) — fiziksel ekran çözünürlüğü
PORTRAIT_CAMERA_VIEW = (720, 405)    # (W, H) — üstteki kamera kutusu (16:9)

# Yön tuşlarının waitKeyEx kodları — backend'e göre değişir (GTK / Qt / Windows).
# waitKey()&0xFF bu kodları ASCII harflere çakıştırıyordu (Yukarı→'R' = kalibrasyon!),
# o yüzden tam kodu waitKeyEx ile okuyup burada karşılaştırıyoruz.
KEY_UP = {65362, 16777235, 2490368, 63232}
KEY_DOWN = {65364, 16777237, 2621440, 63233}

# Dijital zoom (z=yakınlaş, x=uzaklaş). Çarpımsal adım → her basışta yumuşak oran.
# 1.0 = tam görüntü (en uzak), ZOOM_MAX = en yakın. Optik değil, AR görüntüsü kırpılır.
ZOOM_STEP = 1.1
ZOOM_MIN = 1.0
ZOOM_MAX = 6.0

# Saydamlık (ekran [-]/[+] butonları veya +/- tuşları). 0.10–1.00; yüksek = daha şeffaf.
TRANSP_MIN = 0.10
TRANSP_MAX = 1.00
TRANSP_STEP = 0.10
TRANSP_DEFAULT = DEFAULT_TRANSPARENCY  # 0.60

# Dokunmatik buton ızgarası (kameranın altında 2 satır). Her buton = bir komut;
# aynı komutlar klavyeden de gelir (klavye kontrolleri korunur, butonlar EKSTRA).
BUTTON_ROWS = [
    [("row_up", "Sıra ▲"), ("row_down", "Sıra ▼"), ("zoom_in", "Yakınlaş"),
     ("zoom_out", "Uzaklaş"), ("transp_up", "Saydam +"), ("transp_down", "Saydam −")],
    [("direction", "Yön"), ("colorcheck", "Renk"), ("recalibrate", "Kalibre"),
     ("motif", "Motif"), ("quit", "Çıkış")],
]


def ensure_chart(motif: str, rows: int, cols: int, palette: int) -> Path:
    """assets/<motif>_chart.json VARSA olduğu gibi kullan; YOKSA k-means ile üret.

    Kayıtlı chartlar `tools/regen_motif_charts.py` ile motifin GERÇEK ızgarasında
    üretilir (her kağıt karesi = bir dokuma düğümü → piksel kaybı/artifact yok). Bu
    yüzden onları yeniden üretip BOZMAYIZ; oldukları gibi kullanırız. rows/cols/palette
    yalnızca chart hiç yoksa (k-means yedeği) devreye girer. Bir motifin ızgarasını
    değiştirmek istersen regen aracını çalıştır — kalibrasyon ızgaradan bağımsız
    olduğundan her motif kendi gerçek boyutunda kalibre edilen alana sığar.
    """
    assets = Path("assets")
    chart_path = assets / f"{motif}_chart.json"
    if chart_path.exists():
        return chart_path

    src = MOTIFS.get(motif)
    if src is None or not src.exists():
        raise FileNotFoundError(f"motif kaynağı yok: {motif}")
    chart = build_chart(src, rows, cols, palette)
    save_chart(chart, chart_path)  # .preview.png de üretilir
    return chart_path


def _load_chart(motif: str, args) -> Chart:
    """Seçilen motifin chart'ını hazırla/yükle (aynalama yok — kullanıcı isteği)."""
    chart_path = ensure_chart(motif, args.rows, args.cols, args.palette)
    chart = Chart.load(chart_path)
    print(f"chart yüklendi: {chart_path} ({chart.rows}×{chart.cols}, "
          f"{len(chart.palette_rgb)} renk)")
    return chart


def _load_preview(motif: str) -> "np.ndarray | None":
    """Motif seçicide gösterilecek pikselleştirilmiş önizleme (assets/<motif>_chart.preview.png)."""
    p = Path("assets") / f"{motif}_chart.preview.png"
    return cv2.imread(str(p)) if p.exists() else None


def chart_homography(cal_data: dict, chart: Chart):
    """Izgaradan bağımsız birim-kare kalibrasyonunu bu chart'a özgü H'ye çevir.

    cal_data["H_unit_to_cam"] birim kareyi (0..1) kalibre edilen ALANA (kamera
    quad'ı) haritalar. Burada chart-birimini (x∈[0,cols], y∈[0,rows]) birim kareye
    normalize edip (diag(1/cols, 1/rows, 1)) zincirliyoruz → chart [0..cols]×[0..rows]
    TAM olarak kalibre edilen alanı doldurur. Kareler perspektifle eşit olmayabilir
    (sorun değil). rows/cols değişince alan DEĞİŞMEZ; bölünme sıklaşır/seyrekleşir.
    overlay/progress/color_check değişmeden chart-birim H'leriyle çalışmaya devam eder.
    """
    H_unit_to_cam = np.array(cal_data["H_unit_to_cam"], dtype=np.float64)
    norm = np.array(
        [[1.0 / chart.cols, 0.0, 0.0],
         [0.0, 1.0 / chart.rows, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_chart_to_cam = H_unit_to_cam @ norm
    H_cam_to_chart = np.linalg.inv(H_chart_to_cam)
    return H_chart_to_cam, H_cam_to_chart


def pick_motif_interactive(motifs: dict, labels: dict, screen_size, fullscreen=False,
                           default=None) -> str:
    """Kalibrasyondan ÖNCE motif seç: her motif için tıklanabilir kart (önizleme + etiket).

    Tıklanan motifin anahtarını döndürür. ESC → KeyboardInterrupt (iptal). Kartlar
    yatay düzende yan yana, dikey düzende alt alta dizilir; koordinatlar canvas
    (screen_size) uzayında olduğundan WINDOW_NORMAL tıklamalarıyla birebir eşleşir.
    """
    W, H = screen_size
    keys = list(motifs.keys())
    margin, title_h = 30, 70
    landscape = W >= H
    rects = {}
    if landscape:
        card_w = (W - margin * (len(keys) + 1)) // len(keys)
        card_h = H - title_h - 2 * margin
        for i, k in enumerate(keys):
            x = margin + i * (card_w + margin)
            rects[k] = (x, title_h + margin, x + card_w, title_h + margin + card_h)
    else:
        card_w = W - 2 * margin
        card_h = (H - title_h - margin * (len(keys) + 1)) // len(keys)
        for i, k in enumerate(keys):
            y = title_h + margin + i * (card_h + margin)
            rects[k] = (margin, y, margin + card_w, y + card_h)

    previews = {k: _load_preview(k) for k in keys}
    state = {"choice": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            for k, (x1, y1, x2, y2) in rects.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    state["choice"] = k

    win = "MOTIFIKA - Motif Sec"
    setup_window(win, fullscreen, screen_size)  # canvas zaten screen_size → 1:1
    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            canvas = np.full((H, W, 3), 25, dtype=np.uint8)
            texts = [("Motif seç", (margin, 50), 34, (50, 220, 220), True)]
            for k in keys:
                x1, y1, x2, y2 = rects[k]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (45, 45, 45), -1)
                # Varsayılan motifin çerçevesi vurgulu (sarı), diğerleri gri.
                bcol = (50, 220, 220) if k == default else (90, 90, 90)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), bcol, 2)
                # Önizleme: alt ~50px etikete ayrılır, kalanına letterbox ile sığdır.
                ix1, iy1, ix2, iy2 = x1 + 16, y1 + 16, x2 - 16, y2 - 56
                pv = previews.get(k)
                if pv is not None and ix2 > ix1 and iy2 > iy1:
                    fitted = _fit_to_box(pv, ix2 - ix1, iy2 - iy1)
                    canvas[iy1:iy1 + fitted.shape[0], ix1:ix1 + fitted.shape[1]] = fitted
                texts.append((labels.get(k, k), (x1 + 16, y2 - 18), 26, (255, 255, 255), True))
            _draw_texts(canvas, texts)
            cv2.imshow(win, canvas)
            if (cv2.waitKey(20) & 0xFF) == 27:  # ESC → iptal
                raise KeyboardInterrupt("Motif seçimi iptal edildi")
            if state["choice"] is not None:
                return state["choice"]
    finally:
        cv2.destroyWindow(win)


def open_camera_or_image(args, frame_size_hint=(1280, 720), flip: bool = True):
    """Kamera veya sabit görüntü için (frame_provider, cap, boyut) döndür.

    flip=True ise her kare orient_frame ile 180° döndürülür (aynalama yok).
    Çeviri KAYNAKTA yapılır → tracker, renk kontrolü, overlay ve ekran hep aynı
    yönde görür; kalibrasyon da bu yönde toplanır.
    """
    # Sabit görsel modu (test). provider her çağrıda temiz kopya verir.
    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
        if flip:
            img = orient_frame(img)
        return (lambda: img.copy()), None, (img.shape[1], img.shape[0])

    # Canlı kamera modu.
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"kamera açılamadı: {args.camera}")

    # Boyut isteği; kamera desteklemezse sessizce yok sayar → gerçeği geri okuyoruz.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size_hint[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size_hint[1])

    def provider():
        ok, frame = cap.read()
        if not ok:
            return None
        return orient_frame(frame) if flip else frame

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return provider, cap, (real_w, real_h)


def run_calibration_flow(provider, rows, cols, frame_size, out_path: Path,
                         fullscreen: bool = False, orientation: str = "none",
                         screen_size=None) -> dict:
    """4 köşe topla → homography hesapla → JSON'a kaydet.

    orientation, JSON'a yazılır; provider zaten o yöndeki kareyi verdiği için
    köşeler aynı yönde toplanır — homography de bu yönle tutarlı olur.
    screen_size, kalibrasyon penceresini ana ekranla aynı çözünürlüğe kilitler.
    """
    corners, fs = collect_corners_interactive(provider, rows, cols, fullscreen, screen_size)
    if fs is None:  # provider hiç frame vermediyse hint kullan
        fs = frame_size
    # rows/cols yalnızca kalibrasyon ekranındaki bilgi metni içindi; kayıt ızgaradan
    # bağımsız (birim kare → alan). Izgara main.py'de chart'a göre uygulanır.
    return save_calibration(out_path, corners, fs, orientation=orientation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", choices=list(MOTIFS.keys()), default="eli_belinde")
    # Izgara kalibre edilen alanı böler; her motif KENDİ gerçek ızgarasında kayıtlı
    # (regen aracı, native → artifact yok), varsayılan çalışmada o kullanılır.
    # --rows/--cols/--palette YALNIZCA chart hiç yoksa (k-means yedeği) devreye girer.
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=44)
    ap.add_argument("--palette", type=int, default=2)
    ap.add_argument("--direction", choices=["bottom_up", "top_down"], default="bottom_up")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", type=Path, default=None,
                    help="kamera yerine sabit görüntü ile test")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--no-color-check", action="store_true")
    ap.add_argument("--fullscreen", action="store_true",
                    help="MOTIFIKA penceresini tam ekran aç")
    ap.add_argument("--portrait", action="store_true",
                    help="720p portrait ekran düzeni (720×1280): kamera üstte, panel altta")
    ap.add_argument("--no-flip", action="store_true",
                    help="kamerayı döndürme (varsayılan: 180° döndür; aynalama yok)")
    args = ap.parse_args()

    # Varsayılan: kamera karesi 180° döndürülür (orient_frame); aynalama YOK.
    # --no-flip → ham kare. orientation kalibrasyon JSON'una yazılıp uyumda karşılaştırılır
    # (eski "flip" alanı kaldırıldığından eski kalibrasyonlar otomatik geçersiz olur).
    flip = not args.no_flip
    orientation = "rot180" if flip else "none"

    # Donmuş (PyInstaller) çalışırken çalışma dizinini exe'nin yanına al ki
    # assets/, *.jpg/png ve calibration.json gibi GÖRELİ yollar bulunsun/yazılabilsin.
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)

    provider, cap, frame_size = open_camera_or_image(args, flip=flip)

    # Düzen: portrait → kamera üstte (720×405), panel altta tam genişlik (720).
    # Yatay (varsayılan) → kamera solda (880×720), panel sağda (400×720).
    # screen_size motif seçici ve menü ekranlarının çözünürlüğü.
    if args.portrait:
        screen_size = PORTRAIT_SCREEN
        camera_box = PORTRAIT_CAMERA_VIEW
        panel_height = PORTRAIT_SCREEN[1] - PORTRAIT_CAMERA_VIEW[1]  # 1280-405=875
        ui = UIRenderer(panel_width=PORTRAIT_SCREEN[0])
    else:
        screen_size = LANDSCAPE_SCREEN
        camera_box = (LANDSCAPE_SCREEN[0] - LANDSCAPE_PANEL_WIDTH, LANDSCAPE_SCREEN[1])  # (880,720)
        panel_height = LANDSCAPE_SCREEN[1]                                               # 720
        ui = UIRenderer(panel_width=LANDSCAPE_PANEL_WIDTH)

    # Eski kalibrasyon hâlâ uyuyor mu? rows/cols/frame_size + orientation aynıysa kullan.
    # Kalibrasyon MOTİFTEN BAĞIMSIZ → aynı kalibrasyon farklı motiflerle kullanılabilir.
    cal_data = load_calibration(CAL_PATH)
    # Kalibrasyon IZGARADAN BAĞIMSIZ (birim kare → kamera alanı). Eşleşme yalnızca
    # frame_size + orientation'a bakar; rows/cols değişse de AYNI alan kullanılır →
    # ızgara yoğunluğunu değiştirmek yeniden kalibrasyon gerektirmez. "H_unit_to_cam"
    # yoksa dosya eski şemadır → eşleşmez, otomatik yeniden kalibre olur.
    cal_match = (
        cal_data is not None
        and "H_unit_to_cam" in cal_data
        and tuple(cal_data.get("frame_size", [])) == frame_size
        and cal_data.get("orientation") == orientation
    )

    # Ekran (dokunmatik) kontrol durumu. Mouse callback tıklanan butonun KOMUTUNU
    # ctrl["command"]'a yazar; klavye de aynı komutları yazar → tek işleyici (döngü
    # başında) hepsini uygular. transparency/zoom canlı okunur.
    ctrl = {"transparency": TRANSP_DEFAULT, "zoom": 1.0, "command": None, "rects": {}}

    def on_main_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x1, y1, x2, y2) in ctrl["rects"].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                ctrl["command"] = name
                break

    # try/finally: ne olursa olsun (hata/çıkış) kamerayı kapat.
    try:
        # KALİBRASYONDAN ÖNCE: kullanıcı pikselleştirilecek motifi seçer (tıklanabilir
        # kartlar). Motif kalibrasyonu etkilemez → seçim değişse de aynı kalibrasyon olur.
        motif = pick_motif_interactive(MOTIFS, MOTIF_LABELS, screen_size,
                                       args.fullscreen, default=args.motif)
        chart = _load_chart(motif, args)
        direction = args.direction

        if args.recalibrate or not cal_match:
            print("kalibrasyon başlatılıyor: 4 köşeye SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT sırasıyla tıkla")
            cal_data = run_calibration_flow(provider, args.rows, args.cols, frame_size,
                                            CAL_PATH, args.fullscreen, orientation=orientation,
                                            screen_size=screen_size)
            print(f"kalibrasyon kaydedildi: {CAL_PATH}")

        # Izgaradan bağımsız birim-kare kalibrasyonunu bu chart'ın rows×cols'una göre
        # normalize et → chart-birim ↔ kamera. Izgara kalibre edilen alanı tam doldurur.
        H_chart_to_cam, H_cam_to_chart = chart_homography(cal_data, chart)

        # 4 çalışan: döngü boyunca yaşar, her karede metodları çağrılır.
        tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
        renderer = OverlayRenderer(chart, direction=direction, transparency=ctrl["transparency"])
        backend = HSVBackend(chart.palette_rgb)

        do_color_check = not args.no_color_check

        win = "MOTIFIKA"
        setup_window(win, args.fullscreen, screen_size)
        cv2.setMouseCallback(win, on_main_mouse)  # ekran butonları için

        frame_idx = 0
        last_mismatches: list = []  # son kontrol sonucu (ekrandan silmemek için cache)
        t_prev = time.time()
        fps_ema = 0.0

        while True:
            # KOMUT İŞLEYİCİ: ekran butonları (mouse) ve klavye aynı ctrl["command"]'ı
            # yazar → tek yerde uygulanır. Klavye kontrolleri korunur; butonlar ekstra.
            cmd = ctrl["command"]
            ctrl["command"] = None
            if cmd == "quit":
                break
            elif cmd == "motif":
                # Kalibrasyonu KORUYARAK motif değiştir.
                cv2.destroyWindow(win)
                try:
                    chosen = pick_motif_interactive(MOTIFS, MOTIF_LABELS, screen_size,
                                                    args.fullscreen, default=motif)
                except KeyboardInterrupt:
                    chosen = None  # iptal → mevcut motif kalsın
                setup_window(win, args.fullscreen, screen_size)
                cv2.setMouseCallback(win, on_main_mouse)
                if chosen and chosen != motif:
                    motif = chosen
                    chart = _load_chart(motif, args)
                    # Yeni chart'ın rows×cols'u farklı olabilir → H'yi yeniden kur
                    # (kalibrasyon ızgaradan bağımsız, alan aynı kalır).
                    H_chart_to_cam, H_cam_to_chart = chart_homography(cal_data, chart)
                    tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
                    renderer = OverlayRenderer(chart, direction=direction,
                                               transparency=ctrl["transparency"])
                    backend = HSVBackend(chart.palette_rgb)
                    last_mismatches = []
                continue
            elif cmd == "recalibrate":
                cv2.destroyWindow(win)
                cal_data = run_calibration_flow(
                    provider, args.rows, args.cols, frame_size, CAL_PATH,
                    args.fullscreen, orientation=orientation, screen_size=screen_size,
                )
                H_chart_to_cam, H_cam_to_chart = chart_homography(cal_data, chart)
                setup_window(win, args.fullscreen, screen_size)
                cv2.setMouseCallback(win, on_main_mouse)
                continue
            elif cmd == "row_up":
                tracker.bump(-1)
            elif cmd == "row_down":
                tracker.bump(+1)
            elif cmd == "zoom_in":
                ctrl["zoom"] = min(ctrl["zoom"] * ZOOM_STEP, ZOOM_MAX)
            elif cmd == "zoom_out":
                ctrl["zoom"] = max(ctrl["zoom"] / ZOOM_STEP, ZOOM_MIN)
            elif cmd == "transp_up":
                ctrl["transparency"] = min(round(ctrl["transparency"] + TRANSP_STEP, 2), TRANSP_MAX)
            elif cmd == "transp_down":
                ctrl["transparency"] = max(round(ctrl["transparency"] - TRANSP_STEP, 2), TRANSP_MIN)
            elif cmd == "direction":
                direction = "top_down" if direction == "bottom_up" else "bottom_up"
                tracker.direction = direction
                renderer.direction = direction
                tracker.reset_manual()  # offset eski yöne göreydi
            elif cmd == "colorcheck":
                do_color_check = not do_color_check
                if not do_color_check:
                    last_mismatches = []  # cached uyarıları temizle

            frame = provider()
            if frame is None:  # kamera kapandı / görsel sonu
                break
            frame_idx += 1
            renderer.transparency = ctrl["transparency"]  # canlı saydamlık

            # ADIM 1: otomatik atkı cephesi tespiti.
            active_row = tracker.update(frame, H_cam_to_chart)

            # ADIM 2: renk kontrolü (her N karede 1) — bir önceki TAMAMLANMIŞ sırada.
            if do_color_check and frame_idx % COLOR_CHECK_EVERY_N == 0:
                check_row = last_completed_row(active_row, chart.rows, direction)
                if check_row is not None:
                    last_mismatches = check_active_row(
                        frame, H_cam_to_chart, chart, check_row, backend,
                    )
                else:
                    last_mismatches = []

            # ADIM 3: AR overlay → dijital zoom → letterbox.
            ar_view = renderer.render(frame, H_chart_to_cam, active_row)
            ar_view = _zoom_view(ar_view, ctrl["zoom"])
            ar_view = _fit_to_box(ar_view, *camera_box)

            # ADIM 4: panel.
            check_row = last_completed_row(active_row, chart.rows, direction)
            panel = ui.render_panel(
                chart, active_row, direction, last_mismatches,
                height=panel_height if panel_height is not None else ar_view.shape[0],
                check_row=check_row,
            )
            composed = ui.compose(ar_view, panel, vertical=args.portrait)

            # ADIM 5: FPS (EMA ile yumuşatılmış).
            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_ema = 0.85 * fps_ema + 0.15 * inst_fps if fps_ema else inst_fps
            t_prev = t_now

            # ADIM 6: durum satırı + FPS + dokunmatik butonlar → TEK PIL geçişi (net TTF).
            status = f"Saydamlık %{int(round(ctrl['transparency'] * 100))}"
            if ctrl["zoom"] > 1.0:
                status += f"    Zoom x{ctrl['zoom']:.1f}"
            status += f"    Renk: {'Açık' if do_color_check else 'Kapalı'}"
            overlay_texts = [
                (status, (12, 32), 24, (60, 230, 230), True),
                (f"{fps_ema:.0f} FPS", (composed.shape[1] - 12, 32), 22, (60, 230, 230), True, "rs"),
            ]
            rects, btn_texts = _draw_button_bar(composed, camera_box, do_color_check)
            ctrl["rects"] = rects
            _draw_texts(composed, overlay_texts + btn_texts)
            cv2.imshow(win, composed)

            # ADIM 7: klavye → komut (KORUNUR; butonlar ekstra). Komut bir sonraki turda
            # işlenir (göze çarpmaz). waitKeyEx yön tuşlarının tam kodunu verir.
            raw = cv2.waitKeyEx(1)
            key = raw & 0xFF
            if raw in KEY_UP:
                ctrl["command"] = "row_up"
            elif raw in KEY_DOWN:
                ctrl["command"] = "row_down"
            elif key in (27, ord("q")):
                ctrl["command"] = "quit"
            elif key in (ord("r"), ord("R")):
                ctrl["command"] = "recalibrate"
            elif key in (ord("d"), ord("D")):
                ctrl["command"] = "direction"
            elif key in (ord("c"), ord("C")):
                ctrl["command"] = "colorcheck"
            elif key in (ord("+"), ord("=")):
                ctrl["command"] = "transp_up"
            elif key in (ord("-"), ord("_")):
                ctrl["command"] = "transp_down"
            elif key in (ord("z"), ord("Z")):
                ctrl["command"] = "zoom_in"
            elif key in (ord("x"), ord("X")):
                ctrl["command"] = "zoom_out"
    finally:
        # cap None olabilir (sabit görsel modu).
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def _fit_to_box(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Görüntüyü en-boy oranını koruyarak sabit box_w × box_h kutuya oturt (letterbox).

    Çıktı her zaman tam box_w × box_h; sığmayan kenarlar siyahla doldurulur.
    Böylece kamera zoom/çözünürlük değişse de birleşik görüntü boyutu sabit kalır.
    """
    h, w = img.shape[:2]
    # En-boy oranını bozmadan kutuya sığacak en büyük ölçek.
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))

    # Siyah tuval; ölçeklenmiş görüntüyü ortaya yapıştır.
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x0 = (box_w - new_w) // 2
    y0 = (box_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _zoom_view(img: np.ndarray, zoom: float) -> np.ndarray:
    """Görüntünün ORTASINA dijital zoom uygula.

    zoom<=1.0 → değişmez (tam görüntü). zoom>1.0 → merkezdeki 1/zoom'luk bölgeyi
    kırpıp aynı boyuta büyüt. Overlay zaten ar_view'e işlendiğinden kamera ve motif
    birlikte yakınlaşır; homography'ye dokunmaya gerek yok.
    """
    if zoom <= 1.0:
        return img
    h, w = img.shape[:2]
    cw, ch = max(1, round(w / zoom)), max(1, round(h / zoom))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _draw_button_bar(composed: np.ndarray, camera_box, color_check_on: bool) -> tuple:
    """Kamera görüntüsünün altına dokunmatik buton ızgarası çiz.

    Her butona klavyedeki bir komutun karşılığı (BUTTON_ROWS). Rect'ler composed
    koordinatlarında (kamera kutusu composed'da (0,0)'dan başlar) → mouse callback
    ile birebir eşleşir. Dönüş: (rects {komut: rect}, texts [PIL etiketleri]).
    Etiketler tek PIL geçişinde çizilsin diye burada değil, çağırıcıda basılır.
    """
    cam_w, cam_h = camera_box
    bh, gap = 50, 8
    n_rows = len(BUTTON_ROWS)
    y0 = cam_h - (bh * n_rows + gap * (n_rows + 1))
    rects: dict = {}
    texts: list = []
    for ri, row in enumerate(BUTTON_ROWS):
        n = len(row)
        bw = (cam_w - gap * (n + 1)) // n
        y1 = y0 + gap + ri * (bh + gap)
        for ci, (cmd, label) in enumerate(row):
            x1 = gap + ci * (bw + gap)
            x2, y2 = x1 + bw, y1 + bh
            rects[cmd] = (x1, y1, x2, y2)
            on = cmd == "colorcheck" and color_check_on  # aktif mod → yeşil
            cv2.rectangle(composed, (x1, y1), (x2, y2), (60, 95, 60) if on else (55, 55, 55), -1)
            cv2.rectangle(composed, (x1, y1), (x2, y2), (210, 210, 210), 1)
            texts.append((label, (x1 + bw // 2, y1 + bh // 2), 22, (255, 255, 255), True, "mm"))
    return rects, texts


if __name__ == "__main__":
    main()
