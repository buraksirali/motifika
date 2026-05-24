"""MOTİFİKA ana döngü — tüm modülleri sırayla çağırıp birbirine bağlar.

Her karede: kameradan kare al → aktif sırayı bul → (her N karede) renk kontrolü
→ AR overlay çiz → sağ paneli çiz → yan yana yapıştır → ekrana bas → klavye.

Kullanım:
    python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0
    python -m app.main --motif hayat_agaci --rows 50 --cols 30 --image kilim.jpg

Başlangıçta tıklanabilir bir ekranda motif seçilir (kalibrasyondan önce).

Klavye:
    [yukarı ok] vurguyu yukarı taşı [aşağı ok]  vurguyu aşağı taşı
    [z]       yakınlaş               [x]   uzaklaş
    [+]/[-]   saydamlık (daha şeffaf / daha opak)
    [r]       kalibrasyonu sıfırla  [d]   yön değiştir
    [c]       renk kontrolü aç/kapat [q]/ESC  çıkış

Ekran butonları (fare/dokunma): Saydamlik [-]/[+], Motif (kalibrasyonu
koruyarak motif değiştir).
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
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
                         fullscreen: bool = False, orientation: str = "none") -> dict:
    """4 köşe topla → homography hesapla → JSON'a kaydet.

    orientation, JSON'a yazılır; provider zaten o yöndeki kareyi verdiği için
    köşeler aynı yönde toplanır — homography de bu yönle tutarlı olur.
    """
    corners, fs = collect_corners_interactive(provider, rows, cols, fullscreen)
    if fs is None:  # provider hiç frame vermediyse hint kullan
        fs = frame_size
    return save_calibration(out_path, rows, cols, corners, fs, orientation=orientation)


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
    cal_match = (
        cal_data is not None
        and cal_data.get("rows") == args.rows
        and cal_data.get("cols") == args.cols
        and tuple(cal_data.get("frame_size", [])) == frame_size
        and cal_data.get("orientation") == orientation
    )

    # Tıklanabilir ekran kontrollerinin durumu. Mouse callback rect'lere göre günceller;
    # ana döngü bu değerleri okur (saydamlık) / işler (motif değiştir).
    ctrl = {"transparency": TRANSP_DEFAULT, "rects": {}, "switch_motif": False}

    def on_main_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x1, y1, x2, y2) in ctrl["rects"].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "transp_plus":
                    ctrl["transparency"] = min(round(ctrl["transparency"] + TRANSP_STEP, 2), TRANSP_MAX)
                elif name == "transp_minus":
                    ctrl["transparency"] = max(round(ctrl["transparency"] - TRANSP_STEP, 2), TRANSP_MIN)
                elif name == "switch_motif":
                    ctrl["switch_motif"] = True
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
                                            CAL_PATH, args.fullscreen, orientation=orientation)
            print(f"kalibrasyon kaydedildi: {CAL_PATH}")

        H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
        H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)

        # 4 çalışan: döngü boyunca yaşar, her karede metodları çağrılır.
        tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
        renderer = OverlayRenderer(chart, direction=direction, transparency=ctrl["transparency"])
        backend = HSVBackend(chart.palette_rgb)

        do_color_check = not args.no_color_check

        win = "MOTIFIKA"
        _make_window(win, args.fullscreen)
        cv2.setMouseCallback(win, on_main_mouse)  # ekran butonları için

        frame_idx = 0
        last_mismatches: list = []  # son kontrol sonucu (ekrandan silmemek için cache)
        t_prev = time.time()
        fps_ema = 0.0
        zoom = 1.0  # dijital zoom oranı (z/x ile değişir)

        while True:
            # Ekrandaki "Motif" butonu → kalibrasyonu KORUYARAK motif değiştir.
            if ctrl["switch_motif"]:
                ctrl["switch_motif"] = False
                cv2.destroyWindow(win)
                try:
                    chosen = pick_motif_interactive(MOTIFS, MOTIF_LABELS, screen_size,
                                                    args.fullscreen, default=motif)
                except KeyboardInterrupt:
                    chosen = None  # iptal → mevcut motif kalsın
                _make_window(win, args.fullscreen)
                cv2.setMouseCallback(win, on_main_mouse)
                if chosen and chosen != motif:
                    motif = chosen
                    chart = _load_chart(motif, args)
                    tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
                    renderer = OverlayRenderer(chart, direction=direction,
                                               transparency=ctrl["transparency"])
                    backend = HSVBackend(chart.palette_rgb)
                    last_mismatches = []
                continue

            frame = provider()
            if frame is None:  # kamera kapandı / görsel sonu
                break
            frame_idx += 1
            renderer.transparency = ctrl["transparency"]  # ekran [-]/[+] butonlarıyla canlı

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

            # ADIM 3: AR overlay → dijital zoom → sabit kutuya letterbox. Zoom overlay'li
            # görüntüye uygulanır (kamera+motif birlikte yakınlaşır); letterbox sayesinde
            # birleşik görüntü ve panel oranı sabit kalır.
            ar_view = renderer.render(frame, H_chart_to_cam, active_row)
            ar_view = _zoom_view(ar_view, zoom)
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
            # Zoom göstergesi — yalnız yakınlaşmışken (geri bildirim).
            if zoom > 1.0:
                cv2.putText(composed, f"Zoom x{zoom:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ADIM 6: tıklanabilir kontroller (saydamlık [-]/[+] ve Motif) + ekrana bas.
            ctrl["rects"] = _draw_controls(composed, camera_box, ctrl["transparency"])
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
                    args.fullscreen, orientation=orientation,
                )
                H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
                H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)
                _make_window(win, args.fullscreen)
                cv2.setMouseCallback(win, on_main_mouse)
            elif key in (ord("d"), ord("D")):  # yön değiştir (toggle)
                direction = "top_down" if direction == "bottom_up" else "bottom_up"
                tracker.direction = direction
                renderer.direction = direction
                tracker.reset_manual()  # offset eski yöne göreydi
            elif key in (ord("c"), ord("C")):  # renk kontrolü toggle
                do_color_check = not do_color_check
                if not do_color_check:
                    last_mismatches = []  # cached uyarıları temizle
            elif key in (ord("+"), ord("=")):  # saydamlığı artır (daha şeffaf)
                ctrl["transparency"] = min(round(ctrl["transparency"] + TRANSP_STEP, 2), TRANSP_MAX)
            elif key in (ord("-"), ord("_")):  # saydamlığı azalt (daha opak)
                ctrl["transparency"] = max(round(ctrl["transparency"] - TRANSP_STEP, 2), TRANSP_MIN)
            elif key in (ord("z"), ord("Z")):  # yakınlaş
                zoom = min(zoom * ZOOM_STEP, ZOOM_MAX)
            elif key in (ord("x"), ord("X")):  # uzaklaş
                zoom = max(zoom / ZOOM_STEP, ZOOM_MIN)
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


def _btn(img: np.ndarray, rect: tuple, sym: str) -> None:
    """Tek karakterli buton çiz (dolgu + çerçeve + ortalanmış sembol)."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (210, 210, 210), 1)
    cv2.putText(img, sym, (x1 + 13, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)


def _draw_controls(composed: np.ndarray, camera_box, transparency: float) -> dict:
    """Kamera görüntüsünün altına tıklanabilir kontroller çiz, rect sözlüğü döndür.

    Sol alt: Saydamlik [-] %xx [+]. Sağ alt: Motif (değiştir). Rect'ler composed
    (birleşik görüntü) koordinatlarında — kamera kutusu composed'da (0,0)'dan başlar,
    bu yüzden mouse callback ile birebir eşleşir.
    """
    cam_w, cam_h = camera_box
    bh = bw = 46
    by = cam_h - bh - 12
    rects = {}

    # Saydamlık: [-] etiket [+]
    x = 12
    rects["transp_minus"] = (x, by, x + bw, by + bh)
    _btn(composed, rects["transp_minus"], "-")
    label_x, label_w = x + bw + 8, 160
    cv2.rectangle(composed, (label_x, by), (label_x + label_w, by + bh), (35, 35, 35), -1)
    cv2.rectangle(composed, (label_x, by), (label_x + label_w, by + bh), (90, 90, 90), 1)
    cv2.putText(composed, f"Saydamlik %{int(round(transparency * 100))}",
                (label_x + 10, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    plus_x = label_x + label_w + 8
    rects["transp_plus"] = (plus_x, by, plus_x + bw, by + bh)
    _btn(composed, rects["transp_plus"], "+")

    # Motif değiştir (sağ alt).
    mw = 130
    mx = cam_w - 12 - mw
    rects["switch_motif"] = (mx, by, mx + mw, by + bh)
    cv2.rectangle(composed, (mx, by), (mx + mw, by + bh), (60, 60, 60), -1)
    cv2.rectangle(composed, (mx, by), (mx + mw, by + bh), (210, 210, 210), 1)
    cv2.putText(composed, "Motif", (mx + 30, by + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return rects


if __name__ == "__main__":
    main()
