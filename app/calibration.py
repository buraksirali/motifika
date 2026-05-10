# ============================================================================
# DOSYA HAKKINDA
# ============================================================================
"""Çalışma alanı (kilim ROI) kalibrasyonu.

ROI = Region Of Interest = ilgi alanı (kilim sınırları).

Bu dosya ne yapıyor (kısaca):
  Kullanıcı kameradan kilim çalışma alanının 4 köşesine sırayla tıklar
  (SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT). Programdaki "kilim koordinatı"
  ile "kameradaki piksel" arasındaki HOMOGRAPHY matrisi hesaplanıp JSON'a
  kaydedilir. Bu matris bütün diğer modüllerin omurgası.

Kullanım:
    python -m app.calibration --rows 30 --cols 60 --camera 0
    python -m app.calibration --rows 30 --cols 60 --image test.jpg

Çıktı JSON:
    {
        "rows": 30, "cols": 60,
        "camera_corners": [[x,y], ...],         # tıklanan 4 nokta (kamera piksel)
        "frame_size": [w, h],                    # kamera kare boyutu
        "H_chart_to_cam": [[...]],               # 3x3
        "H_cam_to_chart": [[...]]                # 3x3 (ters)
    }
"""

# ============================================================================
# İMPORTLAR
# ============================================================================
# `from __future__ import annotations` — modern tip yazımının eski Python'da
# çalışması için (pattern.py'da detaylı anlatıldı).
from __future__ import annotations

# argparse: komut satırı argümanları (pattern.py'da anlatıldı).
import argparse
# json: chart/kalibrasyon dosyalarını okuma/yazma.
import json
# Path: dosya yolları için akıllı sınıf.
from pathlib import Path

# cv2: OpenCV. Burada özellikle:
#   - VideoCapture (kamera)
#   - namedWindow, imshow, waitKey (UI)
#   - setMouseCallback (mouse olayları)
#   - getPerspectiveTransform (homography hesabı)
#   - circle, putText, polylines (çizimler)
import cv2
# numpy: matris/dizi işlemleri (homography 3×3 matris).
import numpy as np


# ============================================================================
# MODÜL SEVİYESİ SABİTLER
# ============================================================================
# Konvansiyon: ÜST harf = "değişmez" demek (gerçekten Python'da private/const yok,
# sadece geleneksel uyarı: "bu değişmemeli, dokunma").

# 4 köşenin sırası ve gösterilecek metinleri.
# Türkçe karakter YOK çünkü cv2.putText "ş" "ğ" basamaz, kutucuk gösterir.
# Bu sıralama ÇOK ÖNEMLİ: chart_corners() ve diğer fonksiyonlar bu sıraya bağlı.
CORNER_LABELS = ["SOL UST", "SAG UST", "SAG ALT", "SOL ALT"]

# Diğer dosyaların paylaştığı varsayılan kayıt yolu.
# main.py bunu `from app.calibration import DEFAULT_PATH as CAL_PATH` diye okuyor.
DEFAULT_PATH = Path("assets/calibration.json")


# ============================================================================
# FONKSİYON 1: Chart-uzayında 4 köşe noktasını üret
# ============================================================================
def chart_corners(rows: int, cols: int) -> np.ndarray:
    """Chart koordinatlarında köşe noktaları (sol üst sıfır, sağ alt cols x rows)."""
    # Chart-uzayı = kilim haritası. Birim olarak (cols, rows) = bir hücre.
    # 4 köşe sırasıyla:
    #   [0, 0]       = sol üst (x=0, y=0)
    #   [cols, 0]    = sağ üst
    #   [cols, rows] = sağ alt
    #   [0, rows]    = sol alt
    # Bu sıra CORNER_LABELS ile aynı olmak ZORUNDA.
    # Yoksa kullanıcı "sol üst" tıklar, biz "sağ alt" sanırız → matris yamuk olur.
    #
    # `dtype=np.float32`:
    #   cv2.getPerspectiveTransform float32 İSTER, başka tip kabul etmez.
    #   int verirsen exception fırlar.
    #   float64 da OK ama float32 daha hızlı.
    return np.array(
        [[0, 0], [cols, 0], [cols, rows], [0, rows]],
        dtype=np.float32,
    )


# ============================================================================
# FONKSİYON 2: Homography matrislerini hesapla
# ============================================================================
# HOMOGRAPHY NEDİR? (DEEP DIVE)
# Bir DÜZLEMDEKİ noktayı başka düzleme haritalayan 3×3 matris.
# Mesela: kilim düzlemi (chart) ↔ kamera sensörü düzlemi.
#
# Neden "perspektif" diyoruz?
# Çünkü kameranın gözü kilime düz bakmıyor. Hafif yan/yukarı bakıyor.
# Kilimde paralel olan iki çizgi kamerada paralel görünmez (uzaklaştıkça birleşir).
# Affine dönüşüm (sadece kayma+döndürme+ölçek) bunu modelliyemez.
# Perspektif gerek → 3×3 matris.
#
# 3×3 matrisin son hücresi 1 sabit olduğu için 8 bilinmeyen var.
# 4 nokta × 2 koordinat (x,y) = 8 denklem → tam çözülebilir. O yüzden 4 nokta yeter.
def compute_homography(camera_corners: np.ndarray, rows: int, cols: int):

    # KAYNAK noktalar: chart-uzayında 4 köşe (kilim haritasında nereler).
    src_chart = chart_corners(rows, cols)

    # HEDEF noktalar: kamerada nereye düştükleri (kullanıcı tıkladı).
    # `np.asarray` ≠ `np.array`:
    #   np.array() → her zaman KOPYA yapar (yavaş, bellek harcar).
    #   np.asarray() → eğer zaten numpy array ise dokunmaz; değilse array yapar.
    # Burada güvende olmak için asarray + dtype belirtiyoruz.
    dst_cam = np.asarray(camera_corners, dtype=np.float32)

    # `cv2.getPerspectiveTransform(src, dst)` = 4 nokta eşleşmesinden 3×3 matris.
    # ALTERNATIFLER:
    #   getAffineTransform: 3 nokta yeter, perspektif YOK.
    #   findHomography: 4+ nokta, RANSAC ile gürültüye dayanıklı.
    # Biz tam 4 nokta verdiğimiz için getPerspectiveTransform direkt çözümlü.
    H_chart_to_cam = cv2.getPerspectiveTransform(src_chart, dst_cam)

    # `np.linalg.inv` = matris tersi (inverse).
    # Eğer A · x = y ise, x = A⁻¹ · y.
    # H_chart_to_cam, "chart noktasını → kamera pikseline" çeviriyor.
    # Tersi: kamera pikselini → chart noktasına çevirir.
    #
    # ⚠️ Nadiren ters alınamaz (singular matris).
    # Bu, 4 köşeyi KOLİNEAR (aynı çizgide) işaretlersen olur. Pratikte 4 farklı
    # köşe verince hep çalışır.
    #
    # ALTERNATIF: cv2.invert(M) da kullanılabilir, ama np.linalg.inv daha sade.
    H_cam_to_chart = np.linalg.inv(H_chart_to_cam)

    # İki matris döndürüyoruz, çağıran her ikisini de saklayıp kullanır.
    return H_chart_to_cam, H_cam_to_chart


# ============================================================================
# FONKSİYON 3: Kullanıcıdan 4 köşeyi interaktif olarak topla
# ============================================================================
def collect_corners_interactive(frame_provider, rows: int, cols: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pencerede 4 köşeye tıklatır. frame_provider() güncel kareyi döndürür."""

    # NEDEN DICT? (Bu Python tuhaflığı önemli)
    # İçerideki `on_mouse` fonksiyonu state'e YAZMAK istiyor.
    # Python'da nested (iç içe) fonksiyonlar dış değişkene `nonlocal` olmadan YAZAMAZ.
    # AMA mutable bir nesneyi (dict, list) DEĞİŞTİREBİLİR (içine yaz, append vs).
    # O yüzden state'i dict yapıyoruz, içine yazarak workaround yapıyoruz.
    #
    # Eşdeğer çözümler:
    #   (a) `nonlocal points` deklarasyonu + tekil değişken.
    #   (b) Class kullanmak (state'i instance attribute yap).
    # Bu projede dict tercihi sadelik için.
    state = {"points": [], "frozen": None}

    # ----- MOUSE CALLBACK -----
    # OpenCV mouse callback imzası SABİT: 5 parametre almalı.
    # Az/çok argüman koyarsan crash olmaz ama OpenCV her seferinde 5 değer geçer.
    def on_mouse(event, x, y, flags, _):
        #   event: hangi olay? EVENT_LBUTTONDOWN, EVENT_MOUSEMOVE, vs.
        #   x, y: fare pozisyonu (kamera koordinatı, piksel).
        #   flags: ek bilgi (Ctrl/Shift basılı mı, hangi tuş çekili).
        #   _: userdata (callback'e dışarıdan bağlanan ek veri).
        #      Biz kullanmadığımız için `_` ile "umursamıyorum" diyoruz.
        #
        # Sadece SOL TIK ve henüz 4 noktadan azsa kabul et.
        # `EVENT_LBUTTONDOWN` = sol mouse tuşuna BASILDI (bırakıldı değil).
        # `EVENT_LBUTTONUP` = bırakıldı; kullanmıyoruz.
        # 4 noktadan fazlasını görmezden gel (kullanıcı yanlışlıkla beşinciyi tıklarsa).
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 4:
            # Tuple olarak ekle. (x, y) genelde okunması gereken sıra.
            state["points"].append((x, y))

    # Pencerenin başlığı (string).
    win = "MOTIFIKA - Kalibrasyon"

    # `cv2.namedWindow(name, flag)` = bu isimde bir pencere oluştur.
    # Aynı isimde varsa tekrar oluşturmaz, mevcuda referans verir.
    # Flag seçenekleri:
    #   WINDOW_NORMAL    = kullanıcı boyutlandırabilir.
    #   WINDOW_AUTOSIZE  = sabit, resmin boyutunda.
    #   WINDOW_FULLSCREEN = tam ekran.
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Bu pencerede mouse olayı olunca on_mouse fonksiyonunu çağır.
    # 3. parametre userdata (callback'e ek veri); biz kullanmıyoruz, varsayılan None.
    cv2.setMouseCallback(win, on_mouse)

    # frame boyutunu ilk kareyi alınca öğreneceğiz.
    frame_size = None

    # ----- ANA DÖNGÜ -----
    # `while True` = sonsuz döngü. Sadece içeriden `break` veya `raise` ile çıkar.
    while True:

        # ----- Kareyi al -----
        # SPACE'la dondurulmuş bir kare varsa onu kullan, yoksa canlı kareyi al.
        if state["frozen"] is not None:
            # `.copy()` = kopyala. Üzerine çizeceğiz, orijinal frozen kare bozulmasın.
            # Yoksa her döngüde üzerine üzerine çizerdik, daireler birikirdi.
            frame = state["frozen"].copy()
        else:
            # Canlı kareyi al. frame_provider dışarıdan geliyor (kamera veya sabit görsel).
            frame = frame_provider()
            if frame is None:
                # `continue` = döngünün başına dön, alt satırları atla.
                # Kamera henüz hazır değilse atla, sonraki iterasyonda tekrar dene.
                continue
            # frame.shape = (H, W, 3) → cv2 kuralı (W, H) için ters çeviriyoruz.
            # cv2 dünyası HEP (genişlik, yükseklik). numpy HEP (yükseklik, genişlik).
            frame_size = (frame.shape[1], frame.shape[0])

        # ----- Mevcut işaretli noktaları çiz -----
        # `enumerate(liste)` = "indeks ve değeri birlikte ver".
        # for i, p in [(0, (10,20)), (1, (50,80)), ...] gibi.
        # i: 0,1,2,3   p: (x,y) tuple.
        for i, p in enumerate(state["points"]):
            # `cv2.circle(img, merkez, yarıçap, renk, kalınlık)`.
            # `-1` kalınlık = içi DOLU (filled circle).
            # Pozitif sayı: dış çerçeve kalınlığı (içi boş).
            # `(0, 255, 255)` BGR: B=0, G=255, R=255 → SARI (OpenCV BGR!).
            cv2.circle(frame, p, 8, (0, 255, 255), -1)

            # Numarayı yaz.
            cv2.putText(
                frame,
                # `i + 1` = 1, 2, 3, 4 (kullanıcıya 0'dan başlatma, doğal saysın).
                str(i + 1),
                # `(p[0] + 10, p[1] - 10)` = noktanın 10 sağına, 10 yukarısına yaz.
                # Tam üstüne yazsak daireyi kapatırdı.
                (p[0] + 10, p[1] - 10),
                # putText parametre sırası: img, text, origin, font, scale, color, thickness.
                cv2.FONT_HERSHEY_SIMPLEX,  # font tipi (sade SAN-SERIF)
                0.7,                       # ölçek (1.0 standart, 0.7 küçük)
                (0, 255, 255),             # SARI
                2,                         # kalınlık
            )

        # ----- 2+ nokta varsa aralarına çizgi çek -----
        if len(state["points"]) >= 2:
            cv2.polylines(
                frame,
                # `polylines` array LİSTESİ ister, tek array değil. O yüzden [...] sarıyoruz.
                # Bu yapı çoklu polygon çizmek için: [poly1, poly2, ...].
                # Biz tek polygon çizdiğimiz için tek elemanlı liste.
                # `np.int32` çünkü cv2 koordinatları tam sayı bekler.
                [np.array(state["points"], np.int32)],
                # 4 nokta tamamsa polygon kapansın (4. → 1. çizgi de çizilsin).
                # 2-3 nokta varsa açık polygon (line strip).
                isClosed=(len(state["points"]) == 4),
                color=(0, 255, 0),  # YEŞİL (B=0, G=255, R=0).
                thickness=2,
            )

        # ----- Talimat metni -----
        idx = len(state["points"])
        # Python TERNARY OPERATÖR: `A if X else B`.
        # X true ise A, false ise B döner. Tek satır if-else gibi.
        msg = (
            f"Tikla: {CORNER_LABELS[idx]} ({idx + 1}/4)"
            if idx < 4
            else "ENTER=kaydet  R=sifirla  ESC=iptal"
        )
        # Üst sol köşeye kırmızı talimat (BGR'de R=255).
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Altına ızgara bilgisi (beyaz).
        cv2.putText(
            frame, f"Izgara: {rows} sira x {cols} sutun",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        # ----- Kareyi göster -----
        # `cv2.imshow(window_name, image)` = pencereye basar.
        # ZORUNLU: yoksa görüntü güncellenmez. waitKey ile event loop döner.
        cv2.imshow(win, frame)

        # `cv2.waitKey(ms)` = ms milisaniye bekle, basılan tuş kodunu döndür.
        # Bu olmadan pencere donar, OS event loop'u dönmez.
        # 20ms ≈ saniyede 50 frame (gözle yumuşak).
        # `& 0xFF` = son 8 biti al.
        #   Bazı sistemlerde waitKey üst bitleri çöp döner (önişleme bayrakları),
        #   `& 0xFF` ile temizliyoruz. ASCII karakterler 0-127 zaten, sorun olmaz.
        key = cv2.waitKey(20) & 0xFF

        # ESC (ASCII = 27) → iptal et, exception fırlat.
        if key == 27:
            cv2.destroyWindow(win)
            # `raise` = exception fırlat.
            # KeyboardInterrupt = Ctrl+C ile aynı türden bir kesinti.
            # Çağıran taraf `try/except KeyboardInterrupt` ile yakalayabilir.
            raise KeyboardInterrupt("Kalibrasyon iptal edildi")

        # `ord("r")` = 'r' karakterinin ASCII kodu (114).
        # `ord("R")` = 'R' (82).
        # Hem küçük hem büyük R'yi yakalıyoruz (Caps Lock'a karşı dayanıklı).
        if key in (ord("r"), ord("R")):
            # Listeyi BOŞALT (`.clear()` mutate eder, yeni liste yapmaz).
            # `state["points"] = []` da olurdu ama clear daha açık.
            state["points"].clear()
            # Donmayı kaldır.
            state["frozen"] = None

        # SPACE tuşu = ' ' karakteri (ASCII 32).
        # Sadece henüz dondurulmamışken çalışsın (ikinci kez basamaz).
        if key == ord(" ") and state["frozen"] is None:
            # Kareyi dondur. Bundan sonra hep aynı kareye tıklanır.
            # Faydası: eli sallanan kameralarda (örn. webcam) hassas tıklama kolaylaşır.
            state["frozen"] = frame_provider()

        # ENTER: 10 = LF (Linux Enter), 13 = CR (Windows Enter).
        # OS'a göre değişir, ikisini de yakalıyoruz.
        # 4 nokta tamamsa kabul edip çık.
        if key in (10, 13) and len(state["points"]) == 4:
            # `break` = en yakın döngüden çık.
            break

    # Pencereyi kapat.
    cv2.destroyWindow(win)

    # `np.array(liste, dtype=...)` = listeden numpy dizisi yap.
    # Sonuç (4, 2) shape: 4 nokta × 2 koordinat.
    return np.array(state["points"], dtype=np.float32), frame_size


# ============================================================================
# FONKSİYON 4: Kalibrasyonu hesaplayıp JSON'a kaydet
# ============================================================================
def save_calibration(
    out_path: Path,
    rows: int,
    cols: int,
    camera_corners: np.ndarray,
    frame_size: tuple[int, int],
) -> dict:
    # Önce homography hesapla.
    H_chart_to_cam, H_cam_to_chart = compute_homography(camera_corners, rows, cols)

    # Tüm bilgiyi sözlüğe topluyor.
    # `.tolist()` = numpy dizisi → Python listesi (json yazılabilir).
    # `list(frame_size)` = tuple → list (json'da fark etmez ama tutarlılık için).
    data = {
        "rows": rows,
        "cols": cols,
        "camera_corners": camera_corners.tolist(),
        "frame_size": list(frame_size),
        "H_chart_to_cam": H_chart_to_cam.tolist(),
        "H_cam_to_chart": H_cam_to_chart.tolist(),
    }

    # Klasör yoksa oluştur (mkdir -p mantığı).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON yaz, Türkçe düz, 2 boşluk girinti.
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    # Hesaplanan veriyi de geri ver — çağıran tekrar dosyadan okumasın.
    return data


# ============================================================================
# FONKSİYON 5: Diskten kalibrasyon yükle
# ============================================================================
def load_calibration(path: Path = DEFAULT_PATH) -> dict | None:
    # `dict | None` = "ya dict ya None döndürür" (Python 3.10+ syntax).
    # Eski Python'da: `Optional[dict]` (typing modülünden).
    #
    # Dosya yoksa None döndür. Çağıran taraf "kalibrasyon lazım" diye anlar.
    # Exception fırlatmak yerine None tercih edildi: dosya yokluğu BEKLENEN durum.
    if not path.exists():
        return None
    # `path.read_text()` = dosyayı string olarak oku.
    # `json.loads(s)` = string → Python dict.
    return json.loads(path.read_text())


# ============================================================================
# CLI GİRİŞ NOKTASI
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    # Kamera indeksi:
    #   0 = bilgisayardaki ilk kamera (genelde dahili).
    #   1 = ikinci kamera (USB takıldıysa).
    #   2+ = daha fazla.
    # Linux'ta `/dev/video0`, `/dev/video1` ile eşleşir.
    ap.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture indeksi")
    ap.add_argument("--image", type=Path, default=None, help="canli kamera yerine sabit goruntu")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()

    # ----- Sabit görsel modu mu, canlı kamera modu mu? -----
    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)

        # `lambda` = "anonim küçük fonksiyon" (isimsiz fonksiyon).
        # `lambda: img.copy()` = parametresiz, çağrılınca img.copy() döndürür.
        # Eşdeğeri:
        #   def provider():
        #       return img.copy()
        # Tek satır olduğu için lambda tercih edildi.
        # `# noqa: E731` = linter (flake8) "lambda atama yapma" uyarısını yutar.
        provider = lambda: img.copy()  # noqa: E731
    else:
        # `cv2.VideoCapture(index)` = kamerayı aç.
        # cap = "capture" (yakalama) nesnesi.
        cap = cv2.VideoCapture(args.camera)

        # Kamera meşgul, yok veya yetki sorunu varsa açılmaz.
        # Linux'ta /dev/video* yetkisi (genelde video grup üyeliği) gerekir.
        if not cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: {args.camera}")

        # Iç fonksiyon (closure): cap'e referans tutuyor.
        # Her çağrıldığında yeni kare okur.
        def provider():
            # `cap.read()` 2 şey döndürür:
            #   ok: True/False (başarılı mı).
            #   frame: numpy array (kare) veya None.
            ok, frame = cap.read()
            return frame if ok else None

    # ----- Köşeleri topla -----
    # `try/finally` = "ne olursa olsun finally bloğu çalışır".
    # Hata olsa bile, başarılı olsa bile, `break/return` olsa bile.
    # Burada amaç: hata olsa da kamerayı serbest bırak (release).
    try:
        corners, frame_size = collect_corners_interactive(provider, args.rows, args.cols)
    finally:
        # `args.image is None` → canlı kamera modundayız, cap nesnesi var.
        if args.image is None:
            # `cap.release()` = kamerayı SERBEST BIRAK.
            # Yapmazsan kamera meşgul kalır, başka uygulama açamaz.
            # Hatta script bittikten sonra bile bazen kilit kalır (process'i öldürmek gerekir).
            cap.release()

    data = save_calibration(args.out, args.rows, args.cols, corners, frame_size)
    print(f"kalibrasyon kaydedildi: {args.out}")
    print(f"köşeler (kamera px): {data['camera_corners']}")


if __name__ == "__main__":
    main()
