"""Ekran düzeni: SOL = kamera + AR overlay, SAĞ = next-piece paneli + Türkçe uyarı.

Sağ panel (Tetris next-piece tarzı): başlık, sıra sayacı, ilerleme çubuğu,
önceki/şimdiki/sonraki sıra şeritleri, sıradaki renk özeti, hata uyarıları,
klavye yardımı. compose() kamera görüntüsünü panelle yan yana birleştirir.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


PANEL_WIDTH = 480
NEXT_ROW_CELL_PX = 26  # sağdaki şerit hücresi (overlay'den büyük, göz çeksin)

# Şu an KULLANILMIYOR; ileride kullanıcı paletinde Türkçe etiket için.
TURKISH_COLOR_NAMES = {
    "kirmizi": "Kırmızı", "lacivert": "Lacivert", "krem": "Krem",
    "siyah": "Siyah", "beyaz": "Beyaz", "kahverengi": "Kahverengi",
    "yesil": "Yeşil", "sari": "Sarı", "mavi": "Mavi", "bordo": "Bordo",
}


def _color_name(rgb: np.ndarray) -> str:
    """RGB'den yaklaşık Türkçe renk adı (palette küçük, temel ayrım yeterli)."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)

    # Sıralı kategori kuralları — ilk eşleşen kazanır.
    if mx < 60:
        return "Siyah"
    if mn > 200:
        return "Beyaz"
    # Üç kanal birbirine yakın → akromatik (gri/krem).
    if abs(r - g) < 25 and abs(g - b) < 25:
        return "Gri" if mx < 180 else "Krem"
    if r > g and r > b:
        return "Bordo" if r < 140 else "Kırmızı"
    if b > r and b > g:
        return "Lacivert" if b < 140 else "Mavi"
    if g > r and g > b:
        return "Yeşil"
    if r > 150 and g > 100 and b < 80:
        return "Kahverengi"
    return f"({r},{g},{b})"  # kategoriye uymadı — ham değer (debug)


@dataclass
class UIRenderer:
    """Sağ paneli çizen ve kamera görüntüsüyle birleştiren sınıf."""
    panel_width: int = PANEL_WIDTH
    next_row_cell_px: int = NEXT_ROW_CELL_PX

    def render_next_strip(
        self,
        chart,
        active_row: int,
        offset: int,
        direction: str,
        target_w: int,
    ) -> np.ndarray:
        """Aktif sıradan offset kadar uzaktaki sıranın renk şeridi (tam target_w piksel)."""
        # bottom_up'ta sonraki sıra yukarıda (küçük indeks), top_down'da aşağıda.
        if direction == "bottom_up":
            row_idx = active_row - offset
        else:
            row_idx = active_row + offset

        cell_px = self.next_row_cell_px
        h = cell_px

        # Sıra dışı → "(yok)" placeholder.
        if not (0 <= row_idx < chart.rows):
            strip = np.full((h, target_w, 3), 60, dtype=np.uint8)
            cv2.putText(strip, "(yok)", (target_w // 2 - 30, h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return strip

        row_palette_idx = chart.grid[row_idx]
        strip = np.zeros((h, target_w, 3), dtype=np.uint8)

        # _edge(i): i. hücrenin sol kenarı. target_w'yi cols hücreye eşit böler;
        # artan piksel(ler) hücrelere birer birer dağılır (sabit per_cell_w +
        # tek sağ dolgu yönteminin tersine, eşit görüntü verir).
        def _edge(i: int) -> int:
            return i * target_w // chart.cols

        for c, p_idx in enumerate(row_palette_idx):
            strip[:, _edge(c):_edge(c + 1)] = chart.palette_bgr[p_idx]

        # Hücre ayraçları; son kenar target_w'ye denk gelir, 1 px içeri al ki görünsün.
        for c in range(chart.cols + 1):
            x = min(_edge(c), target_w - 1)
            cv2.line(strip, (x, 0), (x, h), (40, 40, 40), 1)
        return strip

    def render_panel(
        self,
        chart,
        active_row: int,
        direction: str,
        mismatches: list,
        height: int,
        check_row: int | None = None,
    ) -> np.ndarray:
        """Sağ kontrol panelini çiz."""
        panel = np.full((height, self.panel_width, 3), 30, dtype=np.uint8)  # koyu gri zemin
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30  # aşağı kayan dikey imleç

        # Başlık.
        cv2.putText(panel, "MOTIFIKA", (12, y), font, 0.9, (50, 220, 220), 2)
        y += 30

        # Sıra sayacı.
        cv2.putText(panel, f"Sira: {active_row} / {chart.rows}",
                    (12, y), font, 0.7, (255, 255, 255), 2)
        y += 25

        # İlerleme oranı: bottom_up'ta active küçüldükçe artar, top_down'da tersi.
        progress = (
            (chart.rows - active_row) / chart.rows
            if direction == "bottom_up"
            else active_row / chart.rows
        )

        # İlerleme çubuğu (arka plan + yeşil dolgu, kalınlık -1 = dolu).
        bar_w = self.panel_width - 24
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 14), (60, 60, 60), -1)
        cv2.rectangle(panel, (12, y), (12 + int(bar_w * progress), y + 14),
                      (50, 200, 50), -1)
        y += 30

        # Önceki / şimdiki / sonraki sıra şeritleri.
        cv2.putText(panel, "Onceki sira", (12, y), font, 0.55, (160, 160, 160), 1)
        y += 8
        prev_strip = self.render_next_strip(chart, active_row, -1, direction, self.panel_width - 24)
        # Slicing assign: paneldeki bölgeyi şeritle override et.
        panel[y:y + prev_strip.shape[0], 12:12 + prev_strip.shape[1]] = prev_strip
        y += prev_strip.shape[0] + 12

        cv2.putText(panel, "Simdiki sira", (12, y), font, 0.6, (50, 220, 255), 2)
        y += 8
        active_strip = self.render_next_strip(chart, active_row, 0, direction, self.panel_width - 24)
        panel[y:y + active_strip.shape[0], 12:12 + active_strip.shape[1]] = active_strip
        y += active_strip.shape[0] + 12

        cv2.putText(panel, "Sonraki sira", (12, y), font, 0.6, (255, 220, 180), 1)
        y += 8
        next1 = self.render_next_strip(chart, active_row, 1, direction, self.panel_width - 24)
        panel[y:y + next1.shape[0], 12:12 + next1.shape[1]] = next1
        y += next1.shape[0] + 16

        # Sonraki sıra renk özeti ("3 Kırmızı, 5 Siyah").
        next_row_idx = (
            active_row - 1 if direction == "bottom_up" else active_row + 1
        )
        if 0 <= next_row_idx < chart.rows:
            counts: dict[int, int] = {}
            for p_idx in chart.grid[next_row_idx]:
                counts[int(p_idx)] = counts.get(int(p_idx), 0) + 1

            parts = []
            # Adet çok olan renk önce (key=-değer ile azalan sıralama).
            for p_idx, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                parts.append(f"{n} {_color_name(chart.palette_rgb[p_idx])}")
            # İlk 3 renk (paneli taşırmasın).
            cv2.putText(panel, "Sonraki: " + ", ".join(parts[:3]),
                        (12, y), font, 0.55, (180, 220, 255), 1)
            y += 24

        # Renk uyarıları.
        if mismatches:
            cv2.putText(panel, f"UYARI ({len(mismatches)} hata)",
                        (12, y), font, 0.7, (60, 60, 240), 2)
            y += 26
            # Kontrol aktif sırada değil bir önceki tamamlanmış sırada yapılıyor.
            row_label = check_row if check_row is not None else active_row
            # En çok 5 hata göster (ekrana sığsın).
            for col, exp_idx, obs_idx, dist in mismatches[:5]:
                exp = _color_name(chart.palette_rgb[exp_idx])
                obs = _color_name(chart.palette_rgb[obs_idx])
                line = f"S{row_label}.{col}: {exp} yerine {obs}"
                cv2.putText(panel, line, (12, y), font, 0.5, (120, 200, 255), 1)
                y += 20
        else:
            cv2.putText(panel, "Renk uyumu: OK",
                        (12, y), font, 0.6, (120, 220, 120), 2)
            y += 24

        # Klavye yardımı — panelin en altına sabit.
        y_help = height - 50
        for line in ["[+/-] sira ayarla", "[r] kalibrasyon", "[q] cikis"]:
            cv2.putText(panel, line, (12, y_help), font, 0.5, (160, 160, 160), 1)
            y_help += 18
        return panel

    def compose(self, camera_view: np.ndarray, panel: np.ndarray) -> np.ndarray:
        """Kamera görüntüsü + paneli yan yana birleştir."""
        h = camera_view.shape[0]
        # Panel yüksekliği kameraya uymuyorsa uydur.
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (panel.shape[1], h))
        return np.hstack([camera_view, panel])
