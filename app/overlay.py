# ============================================================================
# DOSYA HAKKINDA
# ============================================================================
"""AR overlay: chart şablonunu canlı kamera görüntüsüne yansıtır.

AR = Augmented Reality (artırılmış gerçeklik).
Kameranın gördüğü gerçek dünyaya, ekstra "sanal" katmanlar ekliyoruz.

Bu dosya ne yapıyor (kısaca):
  Kullanıcının kilim üzerinde nereyi dokuduğunu görsel olarak anlamasına yardım
  eder. Kameradaki kilim görüntüsünün ÜZERİNE yarı saydam chart yansıtır.

Üç bölge:
  - Tamamlanmış sıralar: düşük opaklık (zarif rehber, dokumayı kapatmasın)
  - Aktif sıra: sarı vurgu çerçevesi (kullanıcı buradasın)
  - Yapılacak sıralar: yüksek opaklık (motif net görünür)
"""

# ============================================================================
# İMPORTLAR
# ============================================================================
from __future__ import annotations

# `dataclass` = boilerplate'siz veri sınıfı.
# Normalde class yazınca __init__, __repr__, __eq__ elle yazılır.
# @dataclass dekoratörü bunları OTOMATIK üretir.
# `field` = özel alan davranışı için yardımcı (örn. init'siz alan, factory default).
from dataclasses import dataclass, field

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================================
# MODÜL SEVİYESİ SABİTLER
# ============================================================================
# Bir hücre kaç piksel? (chart-piksel uzayında)
# Büyütürsen daha keskin chart ama daha fazla bellek (her hücre cell×cell px).
# Küçültürsen daha az detay, daha az bellek.
# 16 = makul denge. Bunu değiştirmek diğer dosyalardaki CELL_PX'lerle senkron olmalı.
CELL_PX = 16

# ALPHA değerleri: 0.0 = tamamen şeffaf (kamera görünür), 1.0 = tamamen opak (chart görünür).
# Üçü de 0.5 yapsan: hepsi yarı yarıya, ayrım kaybolur.
# Üçü de 1.0 yapsan: kameranın hiç görünmez, sadece chart.
DONE_ALPHA = 0.20    # Yapılmış sıralar: çok soluk → dokumanın görünürlüğüne öncelik.
TODO_ALPHA = 0.55    # Yapılacaklar: orta → motif net görünür ama kamera da seçilir.
ACTIVE_ALPHA = 0.65  # Aktif sıra: en vurgulu → "buradasın!".


# ============================================================================
# Chart SINIFI — palette + grid'i bir arada tutar
# ============================================================================
# `@dataclass` = "bu sınıf bir VERİ KABIDIR, otomatik metodlar üret".
# Olmadan elle yazsaydık:
#   class Chart:
#       def __init__(self, rows, cols, palette_rgb, grid, source=""):
#           self.rows = rows; self.cols = cols; ...
#       def __repr__(self): return f"Chart(rows={self.rows}, ...)"
# @dataclass bunu hep üretir, biz sadece alanları yazıyoruz.
@dataclass
class Chart:
    rows: int
    cols: int
    palette_rgb: np.ndarray   # (k, 3) uint8 — RGB renk paleti
    grid: np.ndarray          # (rows, cols) int — palette indeksi
    # 5 alandan biri varsayılan değerli olduğu için EN SONA gelmek ZORUNDA.
    # Python kuralı: varsayılansız alanlardan sonra varsayılanlılar gelir.
    # Mantığı: __init__ pozisyonel argümanları sırayla atar, varsayılansızdan
    # önce varsayılanlı olsa "hangi parametre eksik?" anlaşılmaz.
    source: str = ""

    # ----- COMPUTED PROPERTY -----
    # `@property` = "bu metoda parantezsiz eriş".
    # `chart.palette_bgr` (parantez yok!) yazınca metodu çağırır.
    # Sıradan attribute gibi görünür ama her erişimde hesaplanır.
    #
    # NEDEN depolamak yerine compute?
    # Çünkü palette_rgb'den anlık türev. İkisini birden tutarsak "senkron" sorunu olur:
    # birini güncellersen diğeri geride kalır → bug.
    # Compute güvenli ama her erişimde yeniden hesaplar (bizim için sorun değil, küçük dizi).
    @property
    def palette_bgr(self) -> np.ndarray:
        # `[:, ::-1]` = RGB → BGR (son axis'i ters çevir).
        # OpenCV BGR çalıştığı için cv2 ile çizmeden önce bu lazım.
        return self.palette_rgb[:, ::-1]

    # ----- ALTERNATİF CONSTRUCTOR -----
    # `@classmethod` = "self yerine cls (sınıfın kendisi) alır".
    # Normalde sınıftan instance gerek (chart.metod()), classmethod sınıftan çağrılır
    # (Chart.metod()).
    # Genelde "alternatif yapıcı" pattern'ı için kullanılır:
    #   chart = Chart(rows=..., cols=..., ...)        # normal
    #   chart = Chart.load("path.json")                # classmethod ile
    @classmethod
    def load(cls, path: Path) -> "Chart":
        # Dosyayı oku → JSON parse → sözlüğe dönüştür.
        # `Path(path)` çünkü dışarıdan string de gelebilir, Path'e zorla.
        data = json.loads(Path(path).read_text())
        # `cls(...)` = Chart(...) çağırmak gibi ama sınıf adına bağımsız.
        # Eğer Chart subclass'ı varsa, classmethod o subclass için doğru olur.
        return cls(
            rows=data["rows"],
            cols=data["cols"],
            palette_rgb=np.array(data["palette"], dtype=np.uint8),
            grid=np.array(data["grid"], dtype=int),
            # `data.get("source", "")` = "source varsa al, yoksa boş string".
            # `data["source"]` deseydik ve key yoksa KeyError fırlardı.
            # Eski JSON dosyalarında source yoksa kırılmasın diye `.get`.
            source=data.get("source", ""),
        )


# ============================================================================
# OverlayRenderer — AR yansıtmayı yapan ana sınıf
# ============================================================================
@dataclass
class OverlayRenderer:
    chart: Chart
    cell_px: int = CELL_PX
    direction: str = "bottom_up"
    # `field(init=False)` = bu alan __init__'e parametre OLMAZ.
    # __post_init__ içinde elle dolduracağız.
    # `repr=False` = print(renderer) yapınca yazdırılmasın (büyük array, gereksiz).
    _chart_layer: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Dataclass'ta __init__ otomatik üretiliyor. __init__ bittikten sonra
        # __post_init__ çağrılır (varsa). Burası "ekstra hazırlık" yeri.
        # Burada chart bitmap'ini bir KEZ hazırlıyoruz.
        # NEDEN BİR KEZ? Çünkü ağır hesap, her render'da tekrar yapmak israf.
        self._chart_layer = self._build_chart_layer()

    # ----- _build_chart_layer: chart'ın renkli BITMAP'ini hazırla -----
    def _build_chart_layer(self) -> np.ndarray:
        # `_` ile başlayan metod = "private" (gizli) gelenek.
        # Python'da gerçek private yok ama "_method dışarıdan çağırma" anlamı.
        """Chart pikseli boyutunda, ızgaralı, BGR şablon."""
        cell = self.cell_px

        # FANCY INDEXING (pattern.py'da detayı):
        #   palette_bgr (k, 3), grid (rows, cols).
        #   palette_bgr[grid] → her hücreye o indeksin rengini koyar → (rows, cols, 3).
        layer = self.chart.palette_bgr[self.chart.grid]

        # Her hücreyi cell×cell bloğa şişir.
        # `np.repeat(arr, n, axis)` = axis yönünde n kere tekrarla.
        # İçten dışa: önce satır yönü (axis=0), sonra sütun yönü (axis=1).
        layer = np.repeat(np.repeat(layer, cell, axis=0), cell, axis=1)

        # Belleği yeniden düzenle (kontigü hale getir).
        # `np.repeat` bazen "stride trickery" yaparak gerçek kopya yapmaz.
        # Bu şekilde dönen array bellekte ARDIŞIK olmayabilir.
        # `cv2.line` gibi C kodu ardışık bellek bekler → yoksa hata fırlatır
        # ("Layout of the output array img is incompatible with cv::Mat").
        # `ascontiguousarray` belleği gerçekten ardışık dizip yeni kopya verir.
        # Bu satır olmasaydı bazen patlardı, bazen patlamazdı (CV görsel hatalar).
        layer = np.ascontiguousarray(layer)

        h, w = layer.shape[:2]

        # IZGARA çizgileri (rows+1 yatay çizgi).
        for r in range(self.chart.rows + 1):
            # (40,40,40) BGR = koyu gri. Kalınlık 1 piksel.
            # Yatay çizgi: y sabit (r*cell), x değişiyor (0'dan w'ye).
            cv2.line(layer, (0, r * cell), (w, r * cell), (40, 40, 40), 1)
        # Dikey çizgiler.
        for c in range(self.chart.cols + 1):
            cv2.line(layer, (c * cell, 0), (c * cell, h), (40, 40, 40), 1)

        return layer

    # ----- _alpha_mask: hangi pikselin ne kadar saydam olacağı -----
    def _alpha_mask(self, active_row: int) -> np.ndarray:
        """Her chart pikseline alpha (0..1) ata; aktif sıra çerçevesi dahil."""
        cell = self.cell_px
        # Chart-piksel boyutunda alpha haritası.
        h, w = self.chart.rows * cell, self.chart.cols * cell

        # `np.zeros((h, w), dtype=np.float32)` = TEK KANALLI alpha haritası.
        # 3 değil, çünkü her piksel için TEK saydamlık değeri (RGB için aynı).
        # float32 çünkü 0-1 arası ondalık değerler tutacak.
        alpha = np.zeros((h, w), dtype=np.float32)

        if self.direction == "bottom_up":
            # done = aktif sıranın altı; todo = aktif sıranın üstü; aktif sıra = vurgu
            for r in range(self.chart.rows):
                if r > active_row:
                    # Bottom-up'ta aktif sıranın ALTI = dokunmuş.
                    # Çünkü aşağıdan yukarı dokuma yapıyoruz, alttakiler bitmiş.
                    a = DONE_ALPHA
                elif r == active_row:
                    a = ACTIVE_ALPHA
                else:
                    # r < active_row → üstte = yapılacak.
                    a = TODO_ALPHA
                # `alpha[start:end] = value` = "o satır aralığını value ile doldur".
                # Slicing ile ATAMA → tek değer broadcast edilir.
                # Her sıra cell satırlık (cell_px=16) şeritte aynı alpha'da.
                alpha[r * cell:(r + 1) * cell] = a
        else:
            # top_down: yön tersine.
            for r in range(self.chart.rows):
                if r < active_row:
                    a = DONE_ALPHA
                elif r == active_row:
                    a = ACTIVE_ALPHA
                else:
                    a = TODO_ALPHA
                alpha[r * cell:(r + 1) * cell] = a

        # Aktif sıra çerçevesi (sarı kenarlık) render() içinde direkt çiziliyor (alpha üzerinden değil).
        return alpha

    # ----- render: ASIL İŞ. Her karede çağrılıyor. -----
    def render(
        self,
        frame_bgr: np.ndarray,
        H_chart_to_cam: np.ndarray,
        active_row: int,
    ) -> np.ndarray:
        cell = self.cell_px

        # ----- ÖLÇEK MATRİSİ — chart-piksel ↔ chart-birim arası -----
        # Sorun: chart_layer chart-PİKSEL boyutunda (cols*16 × rows*16).
        #         Ama H_chart_to_cam chart-BİRİM noktalarını (rows, cols cinsinden) bekliyor.
        # Çözüm: önce piksel/16 yap (chart_birim'e dön), sonra H ile çarp.
        #
        # 3×3 matris şu işi yapar:
        #   [x']   [1/cell  0     0]   [x]      [x/cell]
        #   [y'] = [0      1/cell 0] · [y]   =  [y/cell]
        #   [w']   [0      0      1]   [1]      [1     ]
        # Yani "piksel koordinatı" → "birim koordinat".
        scale_inv = np.array(
            [[1 / cell, 0, 0],
             [0, 1 / cell, 0],
             [0, 0, 1]],
            dtype=np.float64,  # float64 daha hassas (matris tersi/çarpımı kümüle olabilir).
        )

        # `@` = MATRIS ÇARPIMI (Python 3.5+).
        # `*` = ELEMENT-WISE çarpım (her hücre kendi çiftiyle).
        # Bunları karıştırma! Matris çarpımı için HER ZAMAN @.
        #
        # M = H_chart_to_cam · scale_inv
        # Yani önce piksel→birim ölçek, sonra birim→kamera dönüşüm.
        # Tek matriste birleştirip warp'a tek geçişte yaptırıyoruz (daha hızlı).
        # chart_px → kamera: H_chart_to_cam (chart_unit→cam) ∘ scale_inv (chart_px→chart_unit)
        M = np.asarray(H_chart_to_cam, dtype=np.float64) @ scale_inv

        # Çıktı boyutu = kamera kare boyutu.
        # (W, H) — cv2 sırası. numpy shape (H, W).
        out_size = (frame_bgr.shape[1], frame_bgr.shape[0])

        # ----- WARP: chart-bitmap'i kameranın perspektifine eğip bük -----
        # `warpPerspective(src, M, dsize, flags)`:
        #   src: kaynak resim
        #   M: 3×3 dönüşüm matrisi
        #   dsize: çıktı boyutu (W, H)
        #   flags: interpolation yöntemi
        # INTER_LINEAR: hızlı, kaliteli ortalama.
        # INTER_NEAREST: hızlı ama tırtıklı (mask warp için OK).
        # INTER_CUBIC: yavaş, daha kaliteli.
        warped_layer = cv2.warpPerspective(self._chart_layer, M, out_size, flags=cv2.INTER_LINEAR)

        # Alpha haritasını da AYNI dönüşümle warp et.
        # Yoksa chart pikselleri ile alpha pikselleri kayar — donuk yerler kaymış chart gösterir.
        alpha = self._alpha_mask(active_row)
        warped_alpha = cv2.warpPerspective(alpha, M, out_size, flags=cv2.INTER_LINEAR)

        # `[..., None]` → ELLIPSIS + None.
        # `...` = "tüm önceki eksenler" (Python Ellipsis nesnesi).
        # `None` = yeni eksen ekle (eşdeğeri np.newaxis).
        # warped_alpha shape: (H, W) → (H, W, 1).
        # Broadcasting için: (H, W, 3) ile çarpılırken son ekseni "yayar".
        warped_alpha = warped_alpha[..., None]

        # ----- ALPHA BLENDING (per-pixel) -----
        # Her piksel için: chart_renk*alpha + kamera_renk*(1-alpha).
        # `\` satır birleştirici, "satır burada bitmedi alt satırda devam".
        # Eşdeğeri: tek satırda yazsak da olurdu, sadece okunurluk için.
        blended = warped_layer.astype(np.float32) * warped_alpha + \
                  frame_bgr.astype(np.float32) * (1.0 - warped_alpha)

        # Floattan tekrar uint8'e. clip ile 0-255 arasında garantile (taşma olursa).
        out = np.clip(blended, 0, 255).astype(np.uint8)

        # En son sarı çerçeveyi çiz, başka her şeyin üstüne (4 köşeli polygon).
        # Bu alpha üzerinden değil, doğrudan çizim.
        self._draw_active_row_border(out, H_chart_to_cam, active_row)
        return out

    # ----- _draw_active_row_border: aktif sıra çerçevesi -----
    def _draw_active_row_border(
        self, frame_bgr: np.ndarray, H_chart_to_cam: np.ndarray, active_row: int,
    ) -> None:
        # Sınır kontrolü. Aktif sıra geçersizse hiç çizme, çık.
        # 0..rows-1 dışındaysa "yok" demektir.
        if not (0 <= active_row < self.chart.rows):
            return

        r = active_row
        # Aktif sıranın 4 köşesi (chart-birim koordinatlarda):
        # sol üst, sağ üst, sağ alt, sol alt.
        pts_chart = np.array(
            [[0, r], [self.chart.cols, r], [self.chart.cols, r + 1], [0, r + 1]],
            dtype=np.float32,
        )

        # ----- HOMOJEN KOORDİNAT trick'i -----
        # 3×3 matris (homography) 2D nokta ile direkt çarpılamaz.
        # 2D noktayı 3D HOMOJEN koordinata çevirmek lazım:
        #   (x, y) → (x, y, 1)
        # Sonra matrisle çarp, son boyuta böl → tekrar 2D.
        #
        # `np.hstack` = yatay birleştir (sütun ekle).
        # Şu an pts_chart shape: (4, 2). Sağına 1'lerden oluşan (4, 1) ekliyoruz.
        # Sonuç: (4, 3) — her satır (x, y, 1).
        pts_h = np.hstack([pts_chart, np.ones((4, 1), dtype=np.float32)])

        # Matris çarpımı:
        # H_chart_to_cam shape: (3, 3)
        # pts_h shape: (4, 3)
        # pts_h.T shape: (3, 4) — transpoze (satır↔sütun).
        # (3,3) @ (3,4) = (3,4)
        # Tekrar `.T` → (4, 3). Yani 4 nokta, her biri (x, y, w) homojen.
        pts_cam = (H_chart_to_cam @ pts_h.T).T

        # Homojenden kartezyene: x'/w', y'/w'.
        # `[:, :2]` = ilk iki sütun (x, y).
        # `[:, 2:3]` = üçüncü sütun (w), boyutu KORU (4,1) → broadcast için.
        # Bölünce (4, 2) / (4, 1) → broadcast → (4, 2).
        pts_cam = pts_cam[:, :2] / pts_cam[:, 2:3]

        # `cv2.polylines` int32 koordinat ister.
        # Sarı kapalı çerçeve, kalınlık 3px (yumuşak vurgu).
        cv2.polylines(
            frame_bgr,
            [pts_cam.astype(np.int32)],
            isClosed=True,
            color=(0, 255, 255),  # SARI (BGR)
            thickness=3,
        )


# ============================================================================
# CLI smoke test — sadece dosyayı tek başına test ederken kullanılır
# ============================================================================
def _cli_smoke():
    """Smoke: chart + kalibrasyon + sahte kamera karesi → overlay PNG.

    "Smoke test" = "duman testi" = "kabaca çalışıyor mu?" testi.
    Asıl programda kullanılmaz, sadece geliştirme sırasında.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=Path("assets/calibration.json"))
    ap.add_argument("--frame", type=Path, default=None, help="kullanılırsa kamera kare olarak okunur")
    ap.add_argument("--active-row", type=int, default=10)
    ap.add_argument("--direction", default="bottom_up")
    ap.add_argument("--out", type=Path, default=Path("/tmp/overlay_test.png"))
    args = ap.parse_args()

    chart = Chart.load(args.chart)
    cal = json.loads(args.calibration.read_text())
    H_chart_to_cam = np.array(cal["H_chart_to_cam"], dtype=np.float64)

    if args.frame:
        frame = cv2.imread(str(args.frame))
    else:
        w, h = cal["frame_size"]
        # `np.full(shape, value)` = istenen şekilde sabit değerli array oluştur.
        # 200 = açık gri (BGR'de 200 her kanal). Sahte kamera karesi simülasyonu.
        frame = np.full((h, w, 3), 200, dtype=np.uint8)

    renderer = OverlayRenderer(chart, direction=args.direction)
    out = renderer.render(frame, H_chart_to_cam, active_row=args.active_row)
    cv2.imwrite(str(args.out), out)
    print(f"yazıldı: {args.out}")


if __name__ == "__main__":
    _cli_smoke()
