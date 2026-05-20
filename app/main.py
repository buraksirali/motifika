"""MOTİFİKA ana döngü — tüm modülleri sırayla çağırıp birbirine bağlar.

Her karede: kameradan kare al → aktif sırayı bul → (her N karede) renk kontrolü
→ AR overlay çiz → sağ paneli çiz → yan yana yapıştır → ekrana bas → klavye.

Kullanım:
    python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0
    python -m app.main --motif hayat_agaci --rows 50 --cols 30 --image kilim.jpg

Klavye:
    [yukarı ok] vurguyu yukarı taşı [aşağı ok]  vurguyu aşağı taşı
    [r]       kalibrasyonu sıfırla  [d]   yön değiştir
    [c]       renk kontrolü aç/kapat [q]/ESC  çıkış
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
    save_calibration,
)
from app.color_check import HSVBackend, check_active_row, last_completed_row
from app.overlay import Chart, OverlayRenderer
from app.pattern import build_chart, save_chart
from app.progress import ProgressTracker
from app.ui import UIRenderer


# Motif kataloğu: isim → kaynak görsel. Yeni motif = buraya satır + görsel ekle.
MOTIFS = {
    "eli_belinde": Path("eli_belinde.jpg"),
    "hayat_agaci": Path("hayat_agaci_ornek.png"),
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


def ensure_chart(motif: str, rows: int, cols: int, palette: int) -> Path:
    """assets/<motif>_chart.json yoksa motif kaynağından üret (varsa cache)."""
    assets = Path("assets")
    chart_path = assets / f"{motif}_chart.json"

    if not chart_path.exists():
        src = MOTIFS.get(motif)
        if src is None or not src.exists():
            raise FileNotFoundError(f"motif kaynağı yok: {motif}")
        chart = build_chart(src, rows, cols, palette)
        save_chart(chart, chart_path)
    return chart_path


def open_camera_or_image(args, frame_size_hint=(1280, 720)):
    """Kamera veya sabit görüntü için (frame_provider, cap, boyut) döndür."""
    # Sabit görsel modu (test). provider her çağrıda temiz kopya verir.
    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
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
        return frame if ok else None

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return provider, cap, (real_w, real_h)


def run_calibration_flow(provider, rows, cols, frame_size, out_path: Path,
                         fullscreen: bool = False) -> dict:
    """4 köşe topla → homography hesapla → JSON'a kaydet."""
    corners, fs = collect_corners_interactive(provider, rows, cols, fullscreen)
    if fs is None:  # provider hiç frame vermediyse hint kullan
        fs = frame_size
    return save_calibration(out_path, rows, cols, corners, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", choices=list(MOTIFS.keys()), default="eli_belinde")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--palette", type=int, default=4)
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
    args = ap.parse_args()

    # Donmuş (PyInstaller) çalışırken çalışma dizinini exe'nin yanına al ki
    # assets/, *.jpg/png ve calibration.json gibi GÖRELİ yollar bulunsun/yazılabilsin.
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)

    # Chart hazırla / yükle.
    chart_path = ensure_chart(args.motif, args.rows, args.cols, args.palette)
    chart = Chart.load(chart_path)
    print(f"chart yüklendi: {chart_path} ({chart.rows}×{chart.cols}, {len(chart.palette_rgb)} renk)")

    provider, cap, frame_size = open_camera_or_image(args)

    # Eski kalibrasyon hâlâ uyuyor mu? rows/cols/frame_size aynıysa kullan.
    # tuple(): JSON'dan list gelir, frame_size tuple — eşitlik için aynı tip olmalı.
    cal_data = load_calibration(CAL_PATH)
    cal_match = (
        cal_data is not None
        and cal_data.get("rows") == args.rows
        and cal_data.get("cols") == args.cols
        and tuple(cal_data.get("frame_size", [])) == frame_size
    )

    # try/finally: ne olursa olsun (hata/çıkış) kamerayı kapat.
    try:
        if args.recalibrate or not cal_match:
            print("kalibrasyon başlatılıyor: 4 köşeye SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT sırasıyla tıkla")
            cal_data = run_calibration_flow(provider, args.rows, args.cols, frame_size,
                                            CAL_PATH, args.fullscreen)
            print(f"kalibrasyon kaydedildi: {CAL_PATH}")

        H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
        H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)

        direction = args.direction

        # 4 çalışan: döngü boyunca yaşar, her karede metodları çağrılır.
        tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
        renderer = OverlayRenderer(chart, direction=direction)
        backend = HSVBackend(chart.palette_rgb)

        # Düzen seçimi: portrait → kamera üstte (720×405), panel altta tam genişlik (720).
        # Yatay (varsayılan) → kamera solda, panel sağda (480 geniş).
        if args.portrait:
            camera_box = PORTRAIT_CAMERA_VIEW
            panel_height = PORTRAIT_SCREEN[1] - PORTRAIT_CAMERA_VIEW[1]  # 1280-405=875
            ui = UIRenderer(panel_width=PORTRAIT_SCREEN[0])
        else:
            camera_box = (LANDSCAPE_SCREEN[0] - LANDSCAPE_PANEL_WIDTH,
                          LANDSCAPE_SCREEN[1])              # (880, 720)
            panel_height = LANDSCAPE_SCREEN[1]              # 720
            ui = UIRenderer(panel_width=LANDSCAPE_PANEL_WIDTH)

        do_color_check = not args.no_color_check

        win = "MOTIFIKA"
        _make_window(win, args.fullscreen)

        frame_idx = 0
        last_mismatches: list = []  # son kontrol sonucu (ekrandan silmemek için cache)
        t_prev = time.time()
        fps_ema = 0.0

        while True:
            frame = provider()
            if frame is None:  # kamera kapandı / görsel sonu
                break
            frame_idx += 1

            # ADIM 1: otomatik atkı cephesi tespiti.
            active_row = tracker.update(frame, H_cam_to_chart)

            # ADIM 2: renk kontrolü (her N karede 1).
            # Aktif sırada değil, bir önceki TAMAMLANMIŞ sırada yapılır
            # (aktif sırada yarım örgüler hatalı sinyal verir).
            if do_color_check and frame_idx % COLOR_CHECK_EVERY_N == 0:
                check_row = last_completed_row(active_row, chart.rows, direction)
                if check_row is not None:
                    last_mismatches = check_active_row(
                        frame, H_cam_to_chart, chart, check_row, backend,
                    )
                else:
                    last_mismatches = []

            # ADIM 3: AR overlay. Sabit kutuya letterbox — kamera zoom/çözünürlük
            # değişse de birleşik görüntü ve panel oranı sabit kalır.
            ar_view = renderer.render(frame, H_chart_to_cam, active_row)
            ar_view = _fit_to_box(ar_view, *camera_box)

            # ADIM 4: panel. Portrait'te sabit yükseklik (875), yatayda kamera kadar.
            check_row = last_completed_row(active_row, chart.rows, direction)
            panel = ui.render_panel(
                chart, active_row, direction, last_mismatches,
                height=panel_height if panel_height is not None else ar_view.shape[0],
                check_row=check_row,
            )
            composed = ui.compose(ar_view, panel, vertical=args.portrait)

            # ADIM 5: FPS (EMA ile yumuşatılmış). max(...,1e-6): sıfıra bölme koruması.
            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_ema = 0.85 * fps_ema + 0.15 * inst_fps if fps_ema else inst_fps
            t_prev = t_now
            cv2.putText(composed, f"{fps_ema:.1f} FPS",
                        (composed.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ADIM 6: ekrana bas.
            cv2.imshow(win, composed)

            # ADIM 7: klavye. waitKeyEx OS event loop'u da işler — olmazsa pencere
            # donar. waitKeyEx (waitKey değil): yön tuşlarının tam kodunu verir.
            raw = cv2.waitKeyEx(1)
            key = raw & 0xFF  # ASCII tuşlar için düşük bayt
            # Ok tuşu = chart'ta görsel yön: sıra 0 üstte, son sıra altta.
            # Yukarı ok → vurgu chart'ta yukarı (sıra index -1), Aşağı ok → aşağı.
            if raw in KEY_UP:
                tracker.bump(-1)
            elif raw in KEY_DOWN:
                tracker.bump(+1)
            elif key in (27, ord("q")):  # ESC / q → çıkış
                break
            elif key in (ord("r"), ord("R")):  # yeniden kalibrasyon
                cv2.destroyWindow(win)
                cal_data = run_calibration_flow(
                    provider, args.rows, args.cols, frame_size, CAL_PATH,
                    args.fullscreen,
                )
                H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
                H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)
                _make_window(win, args.fullscreen)
            elif key in (ord("d"), ord("D")):  # yön değiştir (toggle)
                direction = "top_down" if direction == "bottom_up" else "bottom_up"
                tracker.direction = direction
                renderer.direction = direction
                tracker.reset_manual()  # offset eski yöne göreydi
            elif key in (ord("c"), ord("C")):  # renk kontrolü toggle
                do_color_check = not do_color_check
                if not do_color_check:
                    last_mismatches = []  # cached uyarıları temizle
    finally:
        # cap None olabilir (sabit görsel modu).
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def _make_window(win: str, fullscreen: bool) -> None:
    """MOTIFIKA penceresini oluştur; istenirse tam ekran moduna al.

    WINDOW_NORMAL açılır pencere verir; fullscreen istenince pencere özelliği
    WINDOW_FULLSCREEN'e çekilir. WINDOW_KEEPRATIO: ekran oranı birleşik görüntüden
    farklıysa görüntüyü esnetmeden (letterbox ile) gösterir — yatay düzen 1280×720
    olduğundan 720p ekrana bire bir oturur; farklı çözünürlükte de bozulmaz.
    """
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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


if __name__ == "__main__":
    main()
