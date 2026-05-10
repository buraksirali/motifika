# ============================================================================
# DOSYA HAKKINDA
# ============================================================================
"""Atkı cephesi tespiti — kullanıcının dokuduğu son sıra otomatik bulunur.

ATKI = kilim dokumada yatay (horizontal) renk ipliği. Her sıra bir atkı.
ATKI CEPHESİ = "şu an dokuduğun yer". Cephesi yukarı doğru ilerler.

YAKLAŞIM (algoritma):
  1. Kamera karesini homography ile chart koordinatlarına warp et.
     Çıktı boyutu: (cols * CELL, rows * CELL).
  2. Her sıra için 'dokunmuşluk skoru' hesapla:
     - Dokunmuş satırda renkli iplikler var → yüksek renk varyansı + koyu
     - Dokunmamış satırda yalnızca açık çözgü ipliği görünür → düşük varyans
  3. Skoru tepeden tarayıp dokunmuş→dokunmamış geçiş satırını bul.
  4. EMA + histerezis ile titremeyi azalt; manuel +/- düzeltmesi serbest.

API:
    tracker = ProgressTracker(rows, cols)
    active_row = tracker.update(frame_bgr, H_cam_to_chart)
    tracker.bump(+1) / tracker.bump(-1) / tracker.set(row)
"""

# ============================================================================
# İMPORTLAR
# ============================================================================
from __future__ import annotations

# `dataclass` = sadece veri tutan sınıflar için boilerplate'siz tanım.
from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================================
# MODÜL SEVİYESİ SABİTLER
# ============================================================================
# Burada CELL 12 (overlay 16 idi). Skor hesabı için bu çözünürlük yeter,
# fazlası RAM yer ve hesabı yavaşlatır.
# 30 satır × 60 sütun × 12² = 259,200 piksel. Float32 = 1MB civarı, kabul edilebilir.
CELL_PX = 12

# EMA = Exponential Moving Average (üstel hareketli ortalama).
# Formül: yeni_ortalama = α * yeni_değer + (1-α) * eski_ortalama
# α=0.4 demek: yeni değerin etkisi %40, eskinin %60.
# 1'e yaklaştırırsan (α=0.9 gibi): anlık değer baskın → titreme artar.
# 0'a yaklaştırırsan (α=0.1 gibi): çok ağır → geç tepki verir.
# 0.4 = makul denge.
EMA_ALPHA = 0.4

# En yüksek skorun %55'inden büyükler "dokunmuş" sayılır (relative threshold).
# NEDEN RELATİVE değil ABSOLUTE?
#   Absolute (örn. 0.5): ışık değişince tüm skorlar kayar, yanlış sonuç.
#   Relative: en yüksek hep "dokunmuş" sayılır, ışıktan bağımsız.
# Düşürürsen (0.3): daha fazla sıra "dokunmuş" → yanlış pozitif artar.
# Yükseltirsen (0.8): az sıra dokunmuş → yanlış negatif artar.
THRESH_RATIO = 0.55


# ============================================================================
# ProgressTracker SINIFI — durum tutar (state machine)
# ============================================================================
@dataclass
class ProgressTracker:
    rows: int
    cols: int
    cell_px: int = CELL_PX
    ema_alpha: float = EMA_ALPHA
    thresh_ratio: float = THRESH_RATIO
    direction: str = "bottom_up"  # 'bottom_up' (kilim) | 'top_down'

    def __post_init__(self):
        # 3 İÇ DURUM (state):
        #
        # `_score_ema`: yumuşatılmış skor dizisi (her sıra için bir değer).
        #   None ile başlıyor → ilk update'te dolduracak.
        #   Tip: `np.ndarray | None` (Python 3.10+ syntax).
        #
        # `_active_row`: en son hesaplanan aktif sıra (cache).
        #
        # `_manual_delta`: kullanıcının +/- ile elle eklediği offset.
        #   Otomatik tahmin yapılır + bu offset eklenir.
        #   Reset_manual ile sıfırlanabilir.
        self._score_ema: np.ndarray | None = None
        self._active_row: int = 0
        self._manual_delta: int = 0

    # ----- warp: kamera → chart-piksel uzayına -----
    def warp(self, frame_bgr: np.ndarray, H_cam_to_chart: np.ndarray) -> np.ndarray:
        """Kamera karesini chart koordinatlarına yerleştir (cols*CELL × rows*CELL)."""
        # Bu sefer ÖLÇEK doğru yönde (cell_px ile çarpıyor, BÖLMÜYOR).
        # Çünkü çıktıyı chart-PİKSEL boyutunda istiyoruz (her hücre cell × cell px).
        # H_cam_to_chart kameradan chart-BİRİME (cols, rows skala),
        # sonra ölçek matrisi piksele çıkarıyor.
        scale = np.array(
            [[self.cell_px, 0, 0],
             [0, self.cell_px, 0],
             [0, 0, 1]],
            dtype=np.float64,
        )
        # M = scale · H_cam_to_chart = kameradan → chart-piksele.
        # Tek matrisle warp yapıyoruz, daha hızlı.
        M = scale @ np.asarray(H_cam_to_chart, dtype=np.float64)

        # Çıktı boyutu (W, H) sırasında (cv2 geleneği).
        out_size = (self.cols * self.cell_px, self.rows * self.cell_px)
        return cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)

    # ----- row_scores: her sıra için "dokunmuşluk" puanı -----
    def row_scores(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Her sıra için dokunmuşluk skoru (0..1, yüksek = dokunmuş)."""

        # Renkli → gri tek kanal.
        # `cv2.cvtColor(src, code)` = renk uzayı dönüşümü.
        # COLOR_BGR2GRAY = BGR'yi parlaklığa indirgeme (Y kanalı gibi).
        # NEDEN gri? "Varyans" ve "ortalama parlaklık" hesaplayacağız, renk önemli değil.
        # Tek kanal = 1/3 RAM, 3x hızlı.
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        h = self.cell_px

        # `np.empty(shape, dtype)` = içi BOŞ (rastgele bellek değerleriyle dolu) array.
        # `np.zeros` daha güvenli (sıfırla başlatır) ama yavaş.
        # Hepsini elle dolduracağımız için empty OK (içeriği nasılsa override).
        scores = np.empty(self.rows, dtype=np.float32)

        # Per-row blok: yatay yöndeki varyans (renk farklılığı = doku)
        for r in range(self.rows):
            # O sıranın yatay şeridi: cell_px=12 satır × tüm sütun.
            # `gray[r*h:(r+1)*h]` slicing ile o şeridi seç.
            band = gray[r * h:(r + 1) * h]

            # `band.std()` = standart sapma. Renklerin/parlaklıkların ne kadar dağıldığı.
            # Boş kilim (sadece çözgü) → tek tonda → std≈0.
            # Dokunmuş bölge → çeşitli iplikler → std büyür.
            #
            # `float(...)` çünkü numpy scalar döner, Python float'una çevirip
            # aşağıdaki aritmetik açık olsun (debug kolay).
            std_h = float(band.std())

            # `band.mean()` = ortalama parlaklık 0-255.
            # /255 → 0-1 → 1- → tersi.
            # Parlak (açık çözgü) → düşük darkness, koyu (dokunmuş ip) → yüksek darkness.
            darkness = 1.0 - float(band.mean()) / 255.0

            # Karma skor:
            #   `std_h / 80.0` = normalize (std genelde 0-80 arası → 0-1'e çekiyor).
            #   80 deneysel: tipik kilim std max'i. Adapte edilebilir.
            #   0.6 ve 0.4 = ağırlıklar (toplamı 1).
            # Çeşitlilik daha önemli (0.6), koyuluk yardımcı (0.4).
            scores[r] = 0.6 * std_h / 80.0 + 0.4 * darkness

        # 0-1 arasında sıkıştır (taşmasın).
        scores = np.clip(scores, 0.0, 1.0)
        return scores

    # ----- update: ASIL İŞ. Her karede çağrılır. -----
    def update(self, frame_bgr: np.ndarray, H_cam_to_chart: np.ndarray) -> int:
        warped = self.warp(frame_bgr, H_cam_to_chart)
        scores = self.row_scores(warped)

        # ----- EMA: anlık skoru yumuşat -----
        # İlk kare: önceki yok, direkt skoru kullan (warm start).
        if self._score_ema is None:
            # `.copy()` çünkü direkt referans tutarsak mutate riski.
            self._score_ema = scores.copy()
        else:
            # KLASİK EMA FORMÜLÜ:
            #   yeni_ema = α * yeni_değer + (1-α) * eski_ema
            # α=0.4 demek "yeni %40, eski %60 ağırlıkta".
            # Sonuç: ani değişimler yumuşar, trendler korunur.
            # Numpy elementwise → tüm sıralara aynı formül uygulanır.
            self._score_ema = (
                self.ema_alpha * scores + (1.0 - self.ema_alpha) * self._score_ema
            )

        # ----- En yüksek skor — referans olarak -----
        # `.max()` = dizideki en büyük eleman.
        peak = float(self._score_ema.max())

        # Hiç dokuma yok (boş kilim).
        # 0.05 eşiği: "tüm skorlar çok düşük → boş" yorumu.
        if peak < 0.05:
            # bottom_up'ta başlangıç sırası = en alt = self.rows.
            # top_down'da = en üst = 0.
            auto_row = self.rows if self.direction == "bottom_up" else 0
        else:
            # Eşik = en yüksek skorun %55'i.
            threshold = peak * self.thresh_ratio

            # Boolean dizi: her hücre True (dokunmuş) veya False.
            # numpy comparison vektörize → for döngüsü yok.
            woven = self._score_ema >= threshold

            if self.direction == "bottom_up":
                # alttan yukarı: dokunmuş bölge altta; aktif sıra = en üstteki dokunmuşun bir üstü
                # `np.where(condition)` = "True olan indexleri ver".
                # Sonuç tuple döner (multi-dim için): (array, ).
                # `[0]` = ilk eleman = indeksler array'i.
                idxs = np.where(woven)[0]

                # `idxs.min()` = en küçük dokunmuş indeks (en üstteki dokunmuş).
                # -1 = bir üstü = aktif sıra (bottom_up'ta sıralar yukarı doğru ilerliyor).
                # `if len(idxs)` = "indexler boş değilse" (Pythonic boolean).
                # Hiç dokunmuş yoksa = en alt = rows.
                #
                # PYTHON TERNARY: A if X else B
                #   X true → A, X false → B.
                auto_row = int(idxs.min()) - 1 if len(idxs) else self.rows

                # En az -1 olabilir (alt sınır).
                # `max(a, b)` = ikisinin büyüğü.
                auto_row = max(-1, auto_row)
            else:
                # yukarıdan aşağı: dokunmuş bölge üstte; aktif sıra = ilk dokunmamış
                # `~` = bitwise NOT (boolean array için: True↔False ters çevirme).
                # Yani DOKUNMAMIŞ indeksler.
                idxs = np.where(~woven)[0]
                # En üstteki dokunmamış = aktif (top_down'da yukarıdan aşağıya iniyoruz).
                auto_row = int(idxs.min()) if len(idxs) else self.rows

        # Otomatik bulunana manuel offset'i ekle.
        # `max(a, min(b, x))` = "x'i [a, b] arasına SIKIŞTIR" deyimi (clamp).
        # Eşdeğeri: np.clip(x, a, b).
        self._active_row = max(0, min(self.rows, auto_row + self._manual_delta))
        return self._active_row

    # ----- bump: kullanıcının +/- tuşuna basınca çağrılan -----
    def bump(self, delta: int) -> int:
        # Otomatik tahmin yine yapılır ama bu offset hep eklenir.
        # Yani manuel düzeltme "kalıcı"dır, her update'te otomatiğe eklenir.
        self._manual_delta += delta
        # Aktif satırı da hemen güncelle (UI feedback için).
        self._active_row = max(0, min(self.rows, self._active_row + delta))
        return self._active_row

    # ----- set: belirli satıra zıpla -----
    def set(self, row: int) -> int:
        # Önce sınırla.
        row = max(0, min(self.rows, row))
        # Yeni-eski farkını manuel offset'e ekle.
        # Böylece sonraki update'te otomatik tahmin değişse bile kullanıcının
        # zıplaması korunur.
        self._manual_delta += row - self._active_row
        self._active_row = row
        return self._active_row

    # ----- @property: dışarıdan okumak için -----
    @property
    def active_row(self) -> int:
        # `tracker.active_row` (parantez yok!) yazınca metod çağrılır.
        # Sıradan attribute gibi görünür → API temiz.
        return self._active_row

    # ----- reset_manual: manuel offset'i sıfırla -----
    def reset_manual(self) -> None:
        # Yön değişiminde (d tuşu) eski yön için biriken offset artık geçersiz.
        # Sıfırlamadan yön değiştirsen UI tutarsız davranır.
        self._manual_delta = 0


# ============================================================================
# CLI smoke test
# ============================================================================
def _cli_smoke():
    """Hızlı smoke test: bir görsel + 4 köşe → score grafiği."""
    import argparse, json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("assets/calibration.json"))
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text())
    H_cam_to_chart = np.array(cal["H_cam_to_chart"], dtype=np.float32)

    frame = cv2.imread(str(args.image))
    tracker = ProgressTracker(rows=cal["rows"], cols=cal["cols"])
    row = tracker.update(frame, H_cam_to_chart)
    print(f"aktif sıra: {row} / {cal['rows']}")
    # `np.round(x, n)` = n ondalık basamağa yuvarla.
    # `.tolist()` = numpy → Python list (yazdırırken daha okunaklı).
    print(f"skorlar: {np.round(tracker._score_ema, 2).tolist()}")


if __name__ == "__main__":
    _cli_smoke()
