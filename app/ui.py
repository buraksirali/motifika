# ============================================================================
# DOSYA HAKKINDA
# ============================================================================
"""Ekran düzeni: SOL = kamera + AR overlay, SAĞ = next-piece paneli + Türkçe uyarı.

Bu dosya ne yapıyor:
  Ekranın sağ tarafındaki kontrol panelini çiziyor (Tetris next-piece tarzı).
  Sonra kamera görüntüsünü panelle YAN YANA birleştiriyor.

Panelde ne var:
  - Başlık (MOTIFIKA)
  - "Sıra: 7 / 30" sayacı
  - İlerleme çubuğu (yeşil dolgu)
  - Aktif sıranın renk şeridi
  - Sonraki 2 sıranın renk şeritleri ("next pieces")
  - Sıradaki sıranın renk özeti ("3 Kırmızı, 5 Siyah")
  - Hata uyarıları (varsa)
  - Klavye yardımı
"""

# ============================================================================
# İMPORTLAR
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================================
# MODÜL SEVİYESİ SABİTLER
# ============================================================================
# Panel kaç piksel genişliğinde.
# Daraltırsan panel sıkışır, genişletirsen ekrandan çıkabilir.
# 480 = makul, kamera 1280 ise toplam 1760 → 1080p ekrana sığar.
PANEL_WIDTH = 480

# Sağdaki şeritteki bir hücre ne kadar büyük (overlay'den biraz büyük, göz çeksin).
NEXT_ROW_CELL_PX = 26

# Şu an AKTİF KULLANILMIYOR; ileride kullanıcı paletinde Türkçe etiket için.
# Sözlük: anahtar (key) = ASCII string, değer (value) = Türkçe.
TURKISH_COLOR_NAMES = {
    "kirmizi": "Kırmızı", "lacivert": "Lacivert", "krem": "Krem",
    "siyah": "Siyah", "beyaz": "Beyaz", "kahverengi": "Kahverengi",
    "yesil": "Yeşil", "sari": "Sarı", "mavi": "Mavi", "bordo": "Bordo",
}


# ============================================================================
# Yardımcı fonksiyon: RGB → kabaca Türkçe renk adı
# ============================================================================
def _color_name(rgb: np.ndarray) -> str:
    """Yaklaşık Türkçe renk adı (palette küçük olduğu için temel ayrım yeterli)."""
    # `_` ile başlıyor → "modül-içi yardımcı" (private gelenek).
    # numpy uint8 → Python int (karşılaştırmalar daha temiz).
    # `int(...)` cast'i numpy scalar'ı normal Python int'ine çeviriyor.
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    # En büyük ve en küçük kanal.
    # `max(a, b, c)` üç sayıdan büyüğünü, `min(...)` küçüğünü verir.
    mx, mn = max(r, g, b), min(r, g, b)

    # ----- Sıralı kategori kuralları (ilk eşleşen kazanır) -----

    # Hepsi düşükse koyu/siyah.
    # Yani RGB hepsi <60 → çok az ışık var → siyah ailesi.
    if mx < 60:
        return "Siyah"

    # Hepsi yüksekse açık/beyaz.
    # En düşük kanal bile >200 → tamamı parlak.
    if mn > 200:
        return "Beyaz"

    # Üç kanal birbirine yakınsa gri tonlar (akromatik = renksiz).
    # `abs(r - g)` = mutlak fark.
    # Gri/krem'de R=G=B yaklaşık eşit. <25 fark = "neredeyse aynı".
    # `mx < 180` → koyu gri, yoksa krem (parlak gri).
    if abs(r - g) < 25 and abs(g - b) < 25:
        return "Gri" if mx < 180 else "Krem"

    # R baskınsa kırmızı ailesi. Düşük R = bordo (koyu kırmızı).
    if r > g and r > b:
        return "Bordo" if r < 140 else "Kırmızı"

    # B baskın → mavi ailesi.
    if b > r and b > g:
        return "Lacivert" if b < 140 else "Mavi"

    # G baskın → yeşil.
    if g > r and g > b:
        return "Yeşil"

    # R+G yüksek, B düşük → sıcak orta tonlar = kahverengi.
    # Bu "and" zinciri 3 koşulu birleştirir, hepsi true ise true.
    if r > 150 and g > 100 and b < 80:
        return "Kahverengi"

    # Yukarıdaki kategorilere uymadıysa ham RGB değerini bas (debug için).
    # f-string ile sayı interpolasyonu.
    return f"({r},{g},{b})"


# ============================================================================
# UIRenderer SINIFI — paneli çizen ana sınıf
# ============================================================================
@dataclass
class UIRenderer:
    panel_width: int = PANEL_WIDTH
    next_row_cell_px: int = NEXT_ROW_CELL_PX

    # ----- render_next_strip: bir sıranın renk şeridini çiz -----
    def render_next_strip(
        self,
        chart,
        active_row: int,
        offset: int,
        direction: str,
        target_w: int,
    ) -> np.ndarray:
        """Aktif sıradan offset kadar sonraki sıranın renk şeridi."""
        if direction == "bottom_up":
            # bottom_up: yukarı doğru ilerliyoruz, sonraki yukarıda = küçük indeks.
            # offset=0 aktif sıra, offset=1 bir sonraki, offset=2 ondan sonraki.
            row_idx = active_row - offset
        else:
            # top_down: tersi.
            row_idx = active_row + offset

        cell_px = self.next_row_cell_px
        h = cell_px

        # ----- Sıra dışıysa "yok" placeholder göster -----
        # Mesela aktif sıra zaten 0 ise, "sonraki sıra" diye bir şey yok.
        if not (0 <= row_idx < chart.rows):
            # `np.full(shape, value, dtype)` = istenen şekilde sabit değerli array.
            # 60 = koyu gri (BGR hepsi 60 → koyu placeholder).
            strip = np.full((h, target_w, 3), 60, dtype=np.uint8)
            # Ortaya "(yok)" yaz. (target_w // 2 - 30, h // 2 + 5) yaklaşık merkez.
            cv2.putText(strip, "(yok)", (target_w // 2 - 30, h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return strip

        # ----- Geçerli sıra: renk şeridini çiz -----
        # O sıradaki palet indeksleri (cols uzunluğunda dizi).
        row_palette_idx = chart.grid[row_idx]

        # her hücre cell_px genişliğinde
        # `target_w // chart.cols` = paneli kaç eşit parçaya bölelim.
        # `max(1, ...)` → en az 1 piksel olsun (yoksa 0 olur, görünmez).
        # Mesela target_w=456, cols=60 → per_cell_w = 7.
        per_cell_w = max(1, target_w // chart.cols)

        # Toplam kullanılan genişlik (target_w'nin altında olabilir).
        # 7 * 60 = 420, target_w=456 → 36 piksel boşluk kalır.
        used_w = per_cell_w * chart.cols

        # Boş siyah şerit.
        strip = np.zeros((h, used_w, 3), dtype=np.uint8)

        # `enumerate` = indeks + değer.
        # Her sütun (hücre) için bir renk dikdörtgeni boya.
        for c, p_idx in enumerate(row_palette_idx):
            color_bgr = chart.palette_bgr[p_idx]
            # Slicing assign: `strip[:, x1:x2] = color`.
            # Tüm yükseklik (`:`), o sütunun aralığı, color broadcast ile dolar.
            strip[:, c * per_cell_w:(c + 1) * per_cell_w] = color_bgr

        # Hücre ayraçları (dikey ince çizgiler).
        for c in range(chart.cols + 1):
            x = c * per_cell_w
            cv2.line(strip, (x, 0), (x, h), (40, 40, 40), 1)

        # ----- Tam sığmadıysa sağına dolgu ekle -----
        # used_w < target_w olabilir (per_cell_w yuvarlaması yüzünden).
        # Boşluk dolgusu eklemezsek panel hizalanması bozulur.
        if used_w < target_w:
            pad = np.full((h, target_w - used_w, 3), 30, dtype=np.uint8)
            # `np.hstack([a, b])` = yatay birleştir (yan yana yapıştır).
            # `np.vstack([a, b])` = dikey birleştir (üst üste).
            strip = np.hstack([strip, pad])
        return strip

    # ----- render_panel: ASIL panel çizimi -----
    def render_panel(
        self,
        chart,
        active_row: int,
        direction: str,
        mismatches: list,
        height: int,
        check_row: int | None = None,
    ) -> np.ndarray:
        # Koyu gri arkaplan (BGR=30, hepsi → koyu gri).
        panel = np.full((height, self.panel_width, 3), 30, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # `y` aşağı doğru kaydığımız imleç (cursor).
        # Her metnin altına ekleyeceğiz, böylece dikey akış mantıklı.
        y = 30

        # ----- Başlık -----
        # `cv2.putText(img, text, origin, font, scale, color, thickness)`.
        # Renk BGR (50, 220, 220) ≈ açık turkuaz/cyan.
        cv2.putText(panel, "MOTIFIKA", (12, y), font, 0.9, (50, 220, 220), 2)
        y += 30

        # ----- Sıra sayacı -----
        # f-string ile değişken interpolasyonu.
        # Beyaz "Sira: 7 / 30".
        cv2.putText(panel, f"Sira: {active_row} / {chart.rows}",
                    (12, y), font, 0.7, (255, 255, 255), 2)
        y += 25

        # ----- İlerleme oranı (0-1) hesapla -----
        # bottom_up'ta active küçüldükçe ilerleme artar (alttan yukarı çalışıyoruz).
        # active_row=0 → progress=1 (bittik), active_row=rows → progress=0 (başladık).
        # top_down'da tam tersi.
        progress = (
            (chart.rows - active_row) / chart.rows
            if direction == "bottom_up"
            else active_row / chart.rows
        )

        # Çubuk genişliği (panel - kenarlardan 12+12).
        bar_w = self.panel_width - 24

        # `cv2.rectangle(img, p1, p2, color, thickness)`.
        # `-1` kalınlık = içi DOLU.
        # Pozitif kalınlık: dış çerçeve.
        # p1 = sol-üst köşe, p2 = sağ-alt köşe.
        # Arka plan çubuğu (koyu gri).
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 14), (60, 60, 60), -1)
        # Progress * bar_w kadar yeşil dolgu (üzerine).
        # `int(bar_w * progress)` = kayan noktayı tam sayıya çevir.
        cv2.rectangle(panel, (12, y), (12 + int(bar_w * progress), y + 14),
                      (50, 200, 50), -1)
        y += 30

        # ----- "Aktif sıra" başlık + şerit -----
        cv2.putText(panel, "Aktif sira", (12, y), font, 0.6, (180, 220, 255), 1)
        y += 8
        # offset=0 = aktif sıranın kendisi.
        active_strip = self.render_next_strip(chart, active_row, 0, direction, self.panel_width - 24)

        # SLICING ASSIGN: panele yapıştır.
        # Hedef bölgeyi (y aralığı, x aralığı) seçip array atıyoruz.
        # `panel[y_start:y_end, x_start:x_end] = strip` →
        # numpy o bölgeyi strip'in içeriğiyle override eder.
        panel[y:y + active_strip.shape[0], 12:12 + active_strip.shape[1]] = active_strip
        y += active_strip.shape[0] + 12

        # ----- "Sonraki sıra" -----
        cv2.putText(panel, "Sonraki sira", (12, y), font, 0.6, (255, 220, 180), 1)
        y += 8
        next1 = self.render_next_strip(chart, active_row, 1, direction, self.panel_width - 24)
        panel[y:y + next1.shape[0], 12:12 + next1.shape[1]] = next1
        y += next1.shape[0] + 8

        # ----- "Sonra" (2 sıra ileri) -----
        cv2.putText(panel, "Sonra", (12, y), font, 0.5, (200, 200, 200), 1)
        y += 6
        next2 = self.render_next_strip(chart, active_row, 2, direction, self.panel_width - 24)
        panel[y:y + next2.shape[0], 12:12 + next2.shape[1]] = next2
        y += next2.shape[0] + 16

        # ----- Sıra renk özeti (Tetris next-piece açıklaması) -----
        # Bir sonraki sırayı hesapla.
        next_row_idx = (
            active_row - 1 if direction == "bottom_up" else active_row + 1
        )
        if 0 <= next_row_idx < chart.rows:
            # `dict[int, int]` = anahtar int, değer int olan sözlük.
            # Bu syntax (köşeli parantez) Python 3.9+ tip ipucu.
            # Eski sürümde `Dict[int, int]` (typing modülünden).
            counts: dict[int, int] = {}
            for p_idx in chart.grid[next_row_idx]:
                # `counts.get(key, default)` = "key varsa al, yoksa default ver".
                # Klasik sayma deyimi:
                #   Yoksa 0 al → +1 yap → 1 olur, sözlüğe yaz.
                #   Varsa N al → +1 yap → N+1 olur, sözlüğe yaz.
                # `int(p_idx)` numpy scalar'dan Python int'e (anahtar olarak hashable).
                counts[int(p_idx)] = counts.get(int(p_idx), 0) + 1

            parts = []
            # `sorted(iterable, key=fonksiyon)` = lambda'ya göre sırala.
            # `counts.items()` = [(key1, val1), (key2, val2), ...] çiftleri.
            # `lambda kv: -kv[1]` = "değeri NEGATİF olarak ver" → büyük sayı ÖNCE.
            #   `kv` = (key, value) tuple, `kv[1]` = value.
            #   Negatife çevirmek descending sıralama için yaygın trick.
            for p_idx, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                name = _color_name(chart.palette_rgb[p_idx])
                parts.append(f"{n} {name}")

            # `parts[:3]` = ilk 3 eleman (paneli taşırmasın, ekran sığsın).
            # `", ".join(...)` = listeyi virgül-boşluk ile birleştir → tek string.
            cv2.putText(panel, "Sonraki: " + ", ".join(parts[:3]),
                        (12, y), font, 0.55, (180, 220, 255), 1)
            y += 24

        # ----- Uyarılar (renk uyumsuzlukları) -----
        if mismatches:
            # Listede eleman varsa truthy. Boş liste falsy.
            # Kırmızı (BGR'de R=240 baskın).
            cv2.putText(panel, f"UYARI ({len(mismatches)} hata)",
                        (12, y), font, 0.7, (60, 60, 240), 2)
            y += 26

            # Hangi sıra için uyarı yazıyoruz? check_row varsa o, yoksa active_row.
            # Çünkü kontrol AKTİF sırada değil, BİR ÖNCEKİ tamamlanmış sırada yapılıyor.
            row_label = check_row if check_row is not None else active_row

            # En çok 5 hata göster (ekrana sığsın).
            # `mismatches[:5]` = ilk 5 eleman (eksikse hepsi).
            for col, exp_idx, obs_idx, dist in mismatches[:5]:
                exp = _color_name(chart.palette_rgb[exp_idx])
                obs = _color_name(chart.palette_rgb[obs_idx])
                # "S5.12: Kırmızı yerine Lacivert" gibi bir mesaj.
                line = f"S{row_label}.{col}: {exp} yerine {obs}"
                cv2.putText(panel, line, (12, y), font, 0.5, (120, 200, 255), 1)
                y += 20
        else:
            # Yeşil "OK" — hata yoksa.
            cv2.putText(panel, "Renk uyumu: OK",
                        (12, y), font, 0.6, (120, 220, 120), 2)
            y += 24

        # ----- Klavye yardımı (en alta sabitli) -----
        # Sondan 50 piksel yukarıdan başla.
        # Bu kısım panelin EN ALTINDA (sabit pozisyon), yukarıda kalan boşluğa
        # uyarılar dolar.
        y_help = height - 50
        for line in ["[+/-] sira ayarla", "[r] kalibrasyon", "[q] cikis"]:
            # Açık gri (160, 160, 160), küçük font (0.5).
            cv2.putText(panel, line, (12, y_help), font, 0.5, (160, 160, 160), 1)
            y_help += 18
        return panel

    # ----- compose: kamera + panel = tek görüntü -----
    def compose(self, camera_view: np.ndarray, panel: np.ndarray) -> np.ndarray:
        h = camera_view.shape[0]
        # Panel ile kamera yüksekliği eşit değilse uydur.
        # `cv2.resize` ile genişliği koruyup yüksekliği değiştir.
        # (panel.shape[1], h) = (W, H) → cv2 sırası.
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (panel.shape[1], h))

        # `np.hstack` = horizontal stack (yatay yapıştırma).
        # `np.vstack` = vertical stack (dikey).
        # Her ikisi de aynı dtype ve uyumlu shape ister.
        # Burada her ikisi de uint8 BGR, yatay ekseni eşitlenmiş — sorun yok.
        return np.hstack([camera_view, panel])
