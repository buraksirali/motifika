"""Çalışma alanı (kilim ROI) kalibrasyonu.

Kullanıcı kameradan kilim alanının 4 köşesine sırayla tıklar
(SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT). Chart koordinatı ile kamera pikseli
arasındaki homography matrisi hesaplanıp JSON'a kaydedilir — diğer modüllerin
omurgası.

Kullanım:
    python -m app.calibration --rows 30 --cols 60 --camera 0
    python -m app.calibration --rows 30 --cols 60 --image test.jpg

Çıktı JSON (ızgaradan BAĞIMSIZ — birim kare ↔ kamera):
    {
        "camera_corners": [[x,y], ...],   # tıklanan 4 nokta (kamera piksel)
        "frame_size": [w, h],
        "orientation": "rot180" | "none",
        "H_unit_to_cam": [[...]],          # 3x3: birim kare (0..1) → kamera
        "H_cam_to_unit": [[...]]           # 3x3 (ters)
    }
Izgara (rows×cols) main.py'de chart'a göre uygulanır; kalibre edilen alan sabit
kalır, satır/sütun sayısı yalnızca alanı böler (yoğunluk değişse recalibrate gerekmez).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Köşe sırası. Türkçe karakter yok — cv2.putText "ş"/"ğ" basamaz.
# Bu sıra unit_corners() ve diğer fonksiyonlarla AYNI olmak zorunda.
CORNER_LABELS = ["SOL UST", "SAG UST", "SAG ALT", "SOL ALT"]

# Diğer modüllerin paylaştığı varsayılan kayıt yolu (main.py CAL_PATH olarak okur).
DEFAULT_PATH = Path("assets/calibration.json")


def orient_frame(frame: np.ndarray) -> np.ndarray:
    """Kamera karesini kullanıcı bakışına çevir: 180° döndür (aynalama YOK).

    Kamera projeyi baş aşağı gördüğünden 180° döndürüyoruz. Aynalama bilinçli
    olarak kaldırıldı (kullanıcı isteği). Boyut değişmez (W×H aynı kalır), bu
    yüzden kalibrasyon frame_size'ını bozmaz. Kaynakta uygulanır → tüm hat
    (tracker, renk kontrolü, overlay) aynı yöndedir.
    """
    return cv2.rotate(frame, cv2.ROTATE_180)


def unit_corners() -> np.ndarray:
    """Kalibre edilen alanın 4 köşesi BİRİM karede (0..1), CORNER_LABELS sırasıyla.

    Kalibrasyon artık ızgara satır/sütun sayısından BAĞIMSIZ: 4 tık fiziksel kilim
    ALANINI (birim kare) tanımlar. Izgara (rows×cols) bu alanı yalnızca böler →
    yoğunluk değişse de alan sabit kalır, overlay alanı HEP tam doldurur.
    Sıra CORNER_LABELS ile aynı olmalı; aksi halde matris yamuk çıkar.
    float32: cv2.getPerspectiveTransform başka tip kabul etmez.
    """
    return np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )


def compute_homography(camera_corners: np.ndarray):
    """4 köşeden birim kare ↔ kamera homography matrislerini hesapla (ızgaradan bağımsız).

    Homography = bir düzlemdeki noktayı başka düzleme haritalayan 3x3 matris.
    8 bilinmeyen → 4 nokta × 2 koordinat = 8 denklem, tam çözülebilir. Kaynak =
    birim kare köşeleri (0..1); böylece kaç satır/sütun olduğundan bağımsız tek matris.
    """
    src_unit = unit_corners()
    dst_cam = np.asarray(camera_corners, dtype=np.float32)

    H_unit_to_cam = cv2.getPerspectiveTransform(src_unit, dst_cam)
    # Ters matris: kamera pikseli → birim kare. 4 köşe kolinear değilse hep çalışır.
    H_cam_to_unit = np.linalg.inv(H_unit_to_cam)
    return H_unit_to_cam, H_cam_to_unit


def setup_window(win: str, fullscreen: bool = False,
                 size: "tuple[int, int] | None" = None,
                 keep_ratio: bool = True) -> None:
    """OpenCV penceresi aç; istenirse tam ekran + sabit çözünürlüğe kilitle.

    Projedeki TÜM pencereler (ana ekran, motif seçici, kalibrasyon) bunu kullanır
    ki adımlar arası boyut zıplamasın ve hepsi aynı çözünürlükte açılsın.

    size verilirse pencere tam o piksel boyutuna (resizeWindow) çekilir ve sol-üst
    köşeye (moveWindow 0,0) alınır → çıktı, fiziksel masaüstü çözünürlüğünden
    bağımsız olarak hep size kadar olmaya zorlanır. Pi Touch Screen native
    1280×720 ve composed kare de tam 1280×720 üretildiğinden bu, birebir (1:1)
    render verir. Tam ekran + sabit boyut birlikte: native 1280×720 ekranda boyut
    zaten eşit (resize no-op), farklı ekranda en azından doğru boyutu talep eder.
    keep_ratio (WINDOW_KEEPRATIO): ekran/görüntü oranı farklıysa esnetmeden
    letterbox yapar (güvenlik ağı); kamera karesini esnetmek isteyen kalibrasyon
    penceresi False geçer.
    """
    flags = cv2.WINDOW_NORMAL | (cv2.WINDOW_KEEPRATIO if keep_ratio else 0)
    cv2.namedWindow(win, flags)
    if fullscreen:
        # Tam ekranda SADECE fullscreen property'si — resize/move ÇAĞIRMA.
        # Pi'nin HighGUI (Qt) backend'inde property'den sonra resizeWindow/moveWindow
        # çağırmak fullscreen'i bozup pencereyi normal boyuta indiriyor ve WM onu
        # köşeye yerleştirince görüntü sağ-alta kaymış görünüyordu. Dev (QT5 sistem
        # derlemesi) bu sıralamayı tolere ettiği için sorun yalnızca Pi'de çıkıyordu.
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif size is not None:
        cv2.resizeWindow(win, int(size[0]), int(size[1]))
        cv2.moveWindow(win, 0, 0)


def _letterbox_into(frame: np.ndarray, W: int, H: int):
    """frame'i W×H tuvale en-boy koruyarak yerleştir; (tuval, scale, ox, oy) döndür.

    Kalibrasyonda tıklama TUVAL uzayında alınır; (x-ox)/scale ile ham KAMERA
    pikseline geri çevrilir. Böylece pencere/ekran ölçeklemesinden bağımsız olarak
    "tıklanan yer = tam o nokta" garanti olur. Letterbox matematiği main._fit_to_box
    ile aynıdır ama burada dönüşümü de döndürür (ters çevirebilmek için).
    """
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, ox, oy


def collect_corners_interactive(
    frame_provider, rows: int, cols: int, fullscreen: bool = False,
    screen_size: "tuple[int, int] | None" = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pencerede 4 köşeye tıklatır. frame_provider() güncel kareyi döndürür.

    fullscreen=True ise kalibrasyon penceresi de tam ekran açılır — böylece
    ana pencereyle aynı görünümde kalır, kalibrasyon ekran oranını bozmaz.
    screen_size verilirse pencere o çözünürlüğe kilitlenir (ana ekranla tutarlı).
    """
    # State dict: nested on_mouse nonlocal olmadan mutable nesneyi değiştirebilir.
    state = {"points": [], "frozen": None}

    # OpenCV mouse callback imzası sabit: 5 parametre. (_= kullanılmayan userdata)
    def on_mouse(event, x, y, flags, _):
        # Sol tık + henüz 4'ten az nokta → kabul et.
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 4:
            state["points"].append((x, y))

    win = "MOTIFIKA - Kalibrasyon"
    # keep_ratio=False: gösterilen görüntü (screen_size verilirse) zaten ekranla
    # aynı çözünürlükte bir tuval; native panelde pencere=tuval=ekran → birebir.
    setup_window(win, fullscreen, screen_size, keep_ratio=False)
    cv2.setMouseCallback(win, on_mouse)

    frame_size = None
    # Tuval ← kamera dönüşümü. screen_size yoksa (standalone) birebir kalır.
    scale, ox, oy = 1.0, 0, 0
    while True:
        # SPACE'la dondurulmuş kare varsa onu kullan (sallanan kamerada hassas tık).
        if state["frozen"] is not None:
            frame = state["frozen"].copy()  # üzerine çizeceğiz, orijinali koru
        else:
            frame = frame_provider()
            if frame is None:
                continue  # kamera hazır değil — sonraki iterasyonda dene
            frame_size = (frame.shape[1], frame.shape[0])  # cv2 (W, H)

        # Gösterilecek görüntü: screen_size verildiyse kareyi o çözünürlükte bir
        # tuvale letterbox'la → tıklamalar TUVAL uzayında olur, dönüşümle (scale,
        # ox, oy) ham kamera pikseline birebir çevrilir. Pencere ölçeklemesinden
        # bağımsız tam hizalama. screen_size yoksa ham kareyi göster (birebir).
        if screen_size is not None:
            view, scale, ox, oy = _letterbox_into(frame, screen_size[0], screen_size[1])
        else:
            view, scale, ox, oy = frame, 1.0, 0, 0

        # İşaretli noktaları TUVAL üzerine çiz (tıklanan yere birebir oturur).
        for i, p in enumerate(state["points"]):
            cv2.circle(view, p, 8, (0, 255, 255), -1)
            cv2.putText(
                view, str(i + 1), (p[0] + 10, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        # 2+ nokta varsa aralarına çizgi; 4 nokta tamsa polygon kapat.
        if len(state["points"]) >= 2:
            cv2.polylines(
                view,
                [np.array(state["points"], np.int32)],
                isClosed=(len(state["points"]) == 4),
                color=(0, 255, 0),
                thickness=2,
            )

        # Talimat metni.
        idx = len(state["points"])
        msg = (
            f"Tikla: {CORNER_LABELS[idx]} ({idx + 1}/4)"
            if idx < 4
            else "ENTER=kaydet  R=sifirla  ESC=iptal"
        )
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(
            view, f"Izgara: {rows} sira x {cols} sutun",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow(win, view)
        # & 0xFF: bazı sistemlerde waitKey üst bitleri çöp döner.
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC → iptal
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("Kalibrasyon iptal edildi")

        if key in (ord("r"), ord("R")):  # sıfırla
            state["points"].clear()
            state["frozen"] = None

        if key == ord(" ") and state["frozen"] is None:  # SPACE → kareyi dondur
            state["frozen"] = frame_provider()

        # ENTER: 10 = Linux LF, 13 = Windows CR.
        if key in (10, 13) and len(state["points"]) == 4:
            break

    cv2.destroyWindow(win)
    # Tıklananları (TUVAL uzayı) ham KAMERA piksel uzayına çevir → homography ve
    # overlay ham kamera karesinde çalıştığından köşeler O uzayda olmalı. Kamera
    # sınırına kırp (letterbox şeridine tıklanırsa taşmasın).
    w_cam, h_cam = frame_size if frame_size else (0, 0)
    pts = []
    for x, y in state["points"]:
        cx, cy = (x - ox) / scale, (y - oy) / scale
        if w_cam and h_cam:
            cx = min(max(cx, 0.0), w_cam - 1)
            cy = min(max(cy, 0.0), h_cam - 1)
        pts.append((cx, cy))
    return np.array(pts, dtype=np.float32), frame_size


def save_calibration(
    out_path: Path,
    camera_corners: np.ndarray,
    frame_size: tuple[int, int],
    orientation: str = "none",
) -> dict:
    """Birim-kare homography'sini hesaplayıp JSON'a kaydet, veriyi geri döndür.

    Kalibrasyon IZGARADAN BAĞIMSIZ: yalnızca alan (4 köşe) + frame_size + orientation
    saklanır. rows/cols SAKLANMAZ — ızgara main.py'de chart'a göre uygulanır, böylece
    yoğunluk değişince yeniden kalibrasyon gerekmez ve overlay alanı hep doldurur.

    orientation: köşeler hangi kare yönünde tıklandı? "rot180" (orient_frame ile
    180° döndürülmüş) ya da "none" (ham). main.py cal_match'te karşılaştırır.
    (Eski şema "rows/cols/H_chart_to_cam" yazıyordu; o alanlar artık yok → eski
    dosyalar otomatik geçersiz olup yeniden kalibre olur.)
    """
    H_unit_to_cam, H_cam_to_unit = compute_homography(camera_corners)

    data = {
        "orientation": orientation,
        "camera_corners": camera_corners.tolist(),
        "frame_size": list(frame_size),
        "H_unit_to_cam": H_unit_to_cam.tolist(),
        "H_cam_to_unit": H_cam_to_unit.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def load_calibration(path: Path = DEFAULT_PATH) -> dict | None:
    """Diskten kalibrasyon yükle. Dosya yoksa None (beklenen durum)."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture indeksi")
    ap.add_argument("--image", type=Path, default=None, help="canli kamera yerine sabit goruntu")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--no-flip", action="store_true",
                    help="kamera görüntüsünü çevirme (varsayılan: 180° döndür)")
    args = ap.parse_args()
    flip = not args.no_flip  # main.py ile aynı varsayılan: 180° döndür
    orientation = "rot180" if flip else "none"

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
        base = lambda: img.copy()  # noqa: E731
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: {args.camera}")

        def base():
            ok, frame = cap.read()
            return frame if ok else None

    def provider():
        frame = base()
        return orient_frame(frame) if (flip and frame is not None) else frame

    # try/finally: hata olsa da kamerayı release et (yoksa kilit kalır).
    try:
        corners, frame_size = collect_corners_interactive(provider, args.rows, args.cols)
    finally:
        if args.image is None:
            cap.release()

    data = save_calibration(args.out, corners, frame_size, orientation=orientation)
    print(f"kalibrasyon kaydedildi: {args.out}")
    print(f"köşeler (kamera px): {data['camera_corners']}")


if __name__ == "__main__":
    main()
