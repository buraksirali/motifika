# ============================================================================
# DOSYA HAKKINDA — ANA ORKESTRA ŞEFİ
# ============================================================================
"""MOTİFİKA ana döngü.

Bu dosya = patron. Tüm diğer modülleri sırayla çağırıp birbirine bağlıyor.
Sonsuz döngüde her karede:
  1. Kameradan bir kare al (provider)
  2. Aktif sırayı bul (ProgressTracker)
  3. (Her N karede 1) Renk hatalarını kontrol et (HSVBackend)
  4. Kameranın üstüne rehber çiz (OverlayRenderer)
  5. Sağ paneli çiz (UIRenderer)
  6. Yan yana yapıştır + ekrana bas
  7. Klavye olaylarını yakala (+/- r d c q)

Kullanım:
    python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0
    python -m app.main --motif hayat_agaci --rows 50 --cols 30 --image kilim.jpg
    python -m app.main --recalibrate ...

Klavye:
    [+]/[=]   aktif sırayı +1
    [-]       aktif sırayı -1
    [r]       kalibrasyonu sıfırla (yeni 4-köşe)
    [d]       yön değiştir (bottom_up <-> top_down)
    [c]       renk kontrolünü aç/kapat
    [q]/ESC   çıkış
"""

# ============================================================================
# İMPORTLAR
# ============================================================================
from __future__ import annotations

import argparse
import json
# `time` = saat. FPS hesabı için lazım. time.time() epoch'tan saniye verir.
import time
from pathlib import Path

import cv2
import numpy as np

# ----- Kendi modüllerimizden import -----
# `from app.calibration import (...)` = "app klasöründeki calibration modülünden
# parantez içindekileri al".
from app.calibration import (
    # `as` = takma ad. DEFAULT_PATH'e CAL_PATH adı verdik.
    # Kalan kodda CAL_PATH yazıyoruz, daha açıklayıcı (kalibrasyon olduğunu belli ediyor).
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


# ============================================================================
# SABİTLER
# ============================================================================
# MOTİF KATALOĞU. İsim → kaynak görsel yolu.
# Kullanıcı `--motif eli_belinde` dediğinde bu sözlükten yolu alıyoruz.
# Yeni motif eklemek için: bu sözlüğe yeni satır ekle, ilgili görseli koy.
MOTIFS = {
    "eli_belinde": Path("eli_belinde.jpg"),
    "hayat_agaci": Path("hayat_agaci_ornek.png"),
}

# Her N karede 1 renk kontrolü yap.
#   1 yapsan: her frame, çok yavaşlatır (kontrol pahalı bir iş).
#   30 yapsan: saniyede 1 kez kontrol, gecikir ama hafif.
#   5 = makul denge: göze anlık görünür, CPU'yu yormaz.
# Saniyede ~30 kare ise saniyede 6 kontrol → kullanıcı zaten o hızla dokumuyor.
COLOR_CHECK_EVERY_N = 5


# ============================================================================
# Yardımcı: chart yoksa motif kaynağından üret
# ============================================================================
def ensure_chart(motif: str, rows: int, cols: int, palette: int) -> Path:
    """assets/<motif>_chart.json yoksa motif kaynağından üret."""
    assets = Path("assets")

    # `Path / "string"` = path birleştirme. Path("a") / "b" → Path("a/b").
    # `f"{motif}_chart.json"` = mesela "eli_belinde_chart.json".
    chart_path = assets / f"{motif}_chart.json"

    # Chart yoksa baştan üretelim. Varsa yeniden üretmiyoruz (cache).
    if not chart_path.exists():
        # `dict.get(key)` = key varsa value, yoksa None döndürür.
        src = MOTIFS.get(motif)
        # İki ihtimal: motif tanımsız VEYA görsel dosyası yok.
        if src is None or not src.exists():
            raise FileNotFoundError(f"motif kaynağı yok: {motif}")
        chart = build_chart(src, rows, cols, palette)
        save_chart(chart, chart_path)
    return chart_path


# ============================================================================
# Yardımcı: kamera veya sabit görsel için ortak provider arayüzü
# ============================================================================
def open_camera_or_image(args, frame_size_hint=(1280, 720)):
    """Kamera veya sabit görüntü için frame_provider + cap döndür."""
    # Sabit görsel modu (test için kullanılır).
    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
        # 3 şey döndür: provider (lambda), cap (yok = None), boyut.
        # `lambda` = anonim küçük fonksiyon.
        # `lambda: img.copy()` = parametresiz, çağrılınca img.copy() döner.
        # Eşdeğeri:  def f(): return img.copy()
        # `.copy()` çünkü her seferinde temiz kopya verelim (üzerine çizilse de orijinal bozulmasın).
        # img.shape (H, W, 3) → boyut (W, H) sırasında ver.
        return (lambda: img.copy()), None, (img.shape[1], img.shape[0])

    # ----- Canlı kamera modu -----
    # `cv2.VideoCapture(index_or_url)` = kamera/video dosyası aç.
    # Index: 0 = ilk kamera, 1 = ikinci, vb.
    # String: "video.mp4" gibi yol veya "rtsp://..." gibi URL de olabilir.
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"kamera açılamadı: {args.camera}")

    # Kameraya "şu boyutta görüntü ver" iste.
    # AMA kamera bu boyutu desteklemeyebilir → sessizce yok sayar.
    # O yüzden aşağıda gerçek boyutu geri okuyoruz.
    # `cap.set(prop, value)` = property ayarla.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size_hint[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size_hint[1])

    # Iç fonksiyon (closure): cap'e referans tutuyor.
    # Her çağrıldığında yeni kare okur.
    def provider():
        ok, frame = cap.read()
        return frame if ok else None

    # Gerçek (kameranın verebildiği) boyutu öğren.
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return provider, cap, (real_w, real_h)


# ============================================================================
# Yardımcı: kalibrasyon akışı (4 köşe → JSON)
# ============================================================================
def run_calibration_flow(provider, rows, cols, frame_size, out_path: Path) -> dict:
    corners, fs = collect_corners_interactive(provider, rows, cols)
    # Provider hiç frame vermediyse hint kullan.
    # Sabit görsel modunda zaten frame_size set edilmişti, fallback için.
    if fs is None:
        fs = frame_size
    return save_calibration(out_path, rows, cols, corners, fs)


# ============================================================================
# ANA FONKSİYON — programın kalbi
# ============================================================================
def main():
    # ----- Argümanları tanımla -----
    ap = argparse.ArgumentParser()

    # `choices=[...]` = SADECE bu listedeki değerler kabul edilir.
    # Başka bir şey verirsen argparse hata verip çıkar.
    # `list(MOTIFS.keys())` = ["eli_belinde", "hayat_agaci"] (kataloğumuzdaki anahtarlar).
    ap.add_argument("--motif", choices=list(MOTIFS.keys()), default="eli_belinde")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--palette", type=int, default=4)
    ap.add_argument("--direction", choices=["bottom_up", "top_down"], default="bottom_up")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", type=Path, default=None,
                    help="kamera yerine sabit görüntü ile test")

    # `action="store_true"` = bayrak gibi davranır.
    # `--recalibrate` yazılırsa True olur, yazılmazsa False.
    # Değer almaz (--recalibrate yes gibi bir şey yok).
    # Karşıtı: `action="store_false"` (default True olur, bayrak False yapar).
    ap.add_argument("--recalibrate", action="store_true")

    # `--no-color-check` → args.no_color_check = True.
    # Tireli isimler `_` ile parse edilir (Python değişken adı tireli olamaz).
    ap.add_argument("--no-color-check", action="store_true")

    # `parse_args()` = argümanları gerçekten parse et.
    # Eksik/hatalı argüman varsa burada hata verip program durur.
    args = ap.parse_args()

    # ----- Chart hazırla / yükle -----
    chart_path = ensure_chart(args.motif, args.rows, args.cols, args.palette)
    chart = Chart.load(chart_path)
    print(f"chart yüklendi: {chart_path} ({chart.rows}×{chart.cols}, {len(chart.palette_rgb)} renk)")

    # ----- Kamera/görsel sağlayıcısı -----
    provider, cap, frame_size = open_camera_or_image(args)

    # ----- Eski kalibrasyon hâlâ uyuyor mu? -----
    cal_data = load_calibration(CAL_PATH)
    # Eski kalibrasyon hâlâ uyuyor mu kontrolü:
    # rows/cols/frame_size hepsi aynıysa kullan. Birisi farklıysa baştan kalibre et.
    # `tuple(list)` çünkü JSON'dan list olarak geliyor; frame_size tuple.
    # Eşitlik için aynı tipte olmalı (Python: (1,2) == [1,2] → False).
    cal_match = (
        cal_data is not None
        and cal_data.get("rows") == args.rows
        and cal_data.get("cols") == args.cols
        and tuple(cal_data.get("frame_size", [])) == frame_size
    )

    # ----- ANA İŞ BLOĞU (try/finally ile sarılı) -----
    # try/finally → finally bloğu HER ZAMAN çalışır:
    #   - normal bitiş (return/break)
    #   - exception (raise)
    #   - kullanıcı q'ya basıp çıkış
    # Hangisi olursa olsun kamerayı release etmek istiyoruz.
    try:
        # Gerekiyorsa kalibrasyon yap.
        if args.recalibrate or not cal_match:
            print("kalibrasyon başlatılıyor: 4 köşeye SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT sırasıyla tıkla")
            cal_data = run_calibration_flow(provider, args.rows, args.cols, frame_size, CAL_PATH)
            print(f"kalibrasyon kaydedildi: {CAL_PATH}")

        # JSON listelerini numpy array'e çevir.
        # float64 = en yüksek hassasiyet, matris işlemleri için ideal.
        H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
        H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)

        direction = args.direction

        # ----- 4 ÇALIŞANI KUR -----
        # Bu nesneler döngü boyunca yaşar, her karede metodları çağırılır.
        tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
        renderer = OverlayRenderer(chart, direction=direction)
        ui = UIRenderer()
        backend = HSVBackend(chart.palette_rgb)

        # `not` → tersini al. `--no-color-check` verilmemişse args.no_color_check=False,
        # `not False` → True → renk kontrolü AÇIK.
        do_color_check = not args.no_color_check

        # Pencereyi oluştur.
        win = "MOTIFIKA"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # Döngü değişkenleri.
        frame_idx = 0
        last_mismatches: list = []  # son kontrolün sonucunu ekrandan silmemek için cache.

        # `time.time()` = epoch'tan beri geçen saniye (1970'ten).
        # Float, mikrosaniye hassasiyetli. FPS hesabı için.
        t_prev = time.time()
        fps_ema = 0.0

        # ============================================================================
        # ANA DÖNGÜ — saniyede onlarca kez koşar
        # ============================================================================
        while True:
            # Kameradan/görselden bir kare al.
            frame = provider()
            # Kamera kapanmışsa veya görsel sonu → çık.
            if frame is None:
                break
            frame_idx += 1

            # ----- ADIM 1: Otomatik atkı cephesi tespiti -----
            active_row = tracker.update(frame, H_cam_to_chart)

            # ----- ADIM 2: (her 5 karede 1) Renk kontrolü -----
            # `%` modulo. Sayının kalanı.
            # 5'in katı olduğunda (5, 10, 15, ...) → kalan 0 → True.
            if do_color_check and frame_idx % COLOR_CHECK_EVERY_N == 0:
                # Renk kontrolünü AKTİF sıra üzerinde değil, BİR ÖNCEKİ tamamlanmış sırada yap.
                # Aktif sırada yarım örgüler hatalı sinyal verir.
                check_row = last_completed_row(active_row, chart.rows, direction)
                if check_row is not None:
                    last_mismatches = check_active_row(
                        frame, H_cam_to_chart, chart, check_row, backend,
                    )
                else:
                    # Aktif sıra dokumanın en başındaysa kontrol edilecek bir önceki sıra yok.
                    last_mismatches = []

            # ----- ADIM 3: AR overlay -----
            ar_view = renderer.render(frame, H_chart_to_cam, active_row)
            # Tutarlı UI yüksekliği — kamera farklı çözünürlükte gelse bile.
            # 720 = 720p standart, panel buna göre tasarlandı.
            ar_view = _resize_to_height(ar_view, target_h=720)

            # ----- ADIM 4: Sağ panel -----
            check_row = last_completed_row(active_row, chart.rows, direction)
            panel = ui.render_panel(
                chart, active_row, direction, last_mismatches,
                height=ar_view.shape[0], check_row=check_row,
            )
            composed = ui.compose(ar_view, panel)

            # ----- ADIM 5: FPS hesapla ve göster -----
            t_now = time.time()
            # FPS = 1 / saniye_per_frame.
            # `max(..., 1e-6)` = sıfıra bölünmesin (frame çok hızlı geldiyse).
            # 1e-6 = 0.000001 saniye = 1 mikrosaniye → maksimum 1,000,000 FPS gibi görünür.
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)

            # FPS göstergesi için EMA (titremesin).
            # İlk frame'de fps_ema=0, doğrudan inst kullan (warm start).
            # Sonrasında: %85 eski + %15 yeni → çok yumuşak.
            fps_ema = 0.85 * fps_ema + 0.15 * inst_fps if fps_ema else inst_fps
            t_prev = t_now

            # FPS metnini sağ üst köşeye yaz.
            cv2.putText(composed, f"{fps_ema:.1f} FPS",
                        # `composed.shape[1]` = toplam genişlik. -130 = sağdan 130 piksel içeride.
                        (composed.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ----- ADIM 6: Ekrana bas -----
            cv2.imshow(win, composed)

            # ----- ADIM 7: Klavye -----
            # 1ms bekle = mümkün olduğunca yüksek FPS.
            # waitKey OS event loop'u da işliyor, olmazsa pencere donar.
            key = cv2.waitKey(1) & 0xFF

            # `if/elif/else` zinciri = "ilk eşleşen kazanır".
            if key in (27, ord("q")):
                # ESC (27) veya q → çıkış.
                break
            elif key in (ord("+"), ord("=")):
                # = tuşu shift'siz +'ya denk gelir (Türkçe klavyede de pratik).
                tracker.bump(+1)
            elif key in (ord("-"), ord("_")):
                tracker.bump(-1)
            elif key in (ord("r"), ord("R")):
                # Yeniden kalibrasyon.
                cv2.destroyWindow(win)
                cal_data = run_calibration_flow(
                    provider, args.rows, args.cols, frame_size, CAL_PATH,
                )
                # Yeni matrisleri yükle.
                H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
                H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)
                # Pencereyi yeniden aç.
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            elif key in (ord("d"), ord("D")):
                # Yön değiştir (toggle).
                # Ternary: "bottom_up" ise "top_down" yap, değilse "bottom_up".
                direction = "top_down" if direction == "bottom_up" else "bottom_up"
                tracker.direction = direction
                renderer.direction = direction
                # Manuel offset eski yöne göreydi, sıfırla (anlamı kalmadı).
                tracker.reset_manual()
            elif key in (ord("c"), ord("C")):
                # Renk kontrolünü aç/kapat (toggle).
                # `not bool` = tersi.
                do_color_check = not do_color_check
                if not do_color_check:
                    # Kapanırsa eski uyarılar ekrandan silinsin (cached olanı temizle).
                    last_mismatches = []
    finally:
        # `try/finally` = ne olursa olsun (hata bile) kamerayı kapat.
        # cap None olabilir (sabit görsel modu); bu durumda dokunma.
        if cap is not None:
            cap.release()
        # Tüm pencereleri kapat (cv2 leak yapmasın).
        cv2.destroyAllWindows()


# ============================================================================
# Yardımcı: resmin yüksekliğini hedef yüksekliğe çek
# ============================================================================
def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    # Zaten istenen yükseklik → kopya/işlem yok (early exit).
    if h == target_h:
        return img
    # Yeni genişlik = w * (target_h / h). En-boy oranını korur.
    # Mesela 1080×1920 → target_h=720:
    #   scale = 720/1080 = 0.667
    #   new_w = 1920 * 0.667 = 1280
    #   sonuç: 720×1280 (16:9 oranı korunur).
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


# ============================================================================
# CLI giriş noktası
# ============================================================================
# `__name__ == "__main__"` deyimi pattern.py'da detaylı anlatıldı.
# Özet: dosya doğrudan çalıştırılırsa main(), import edilirse koşmaz.
if __name__ == "__main__":
    main()
