"""Ekran düzeni: SOL = kamera + AR overlay, SAĞ = next-piece paneli + Türkçe uyarı."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


PANEL_WIDTH = 480
NEXT_ROW_CELL_PX = 26
TURKISH_COLOR_NAMES = {
    "kirmizi": "Kırmızı", "lacivert": "Lacivert", "krem": "Krem",
    "siyah": "Siyah", "beyaz": "Beyaz", "kahverengi": "Kahverengi",
    "yesil": "Yeşil", "sari": "Sarı", "mavi": "Mavi", "bordo": "Bordo",
}


def _color_name(rgb: np.ndarray) -> str:
    """Yaklaşık Türkçe renk adı (palette küçük olduğu için temel ayrım yeterli)."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 60:
        return "Siyah"
    if mn > 200:
        return "Beyaz"
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
    return f"({r},{g},{b})"


@dataclass
class UIRenderer:
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
        """Aktif sıradan offset kadar sonraki sıranın renk şeridi."""
        if direction == "bottom_up":
            row_idx = active_row - offset
        else:
            row_idx = active_row + offset
        cell_px = self.next_row_cell_px
        h = cell_px

        if not (0 <= row_idx < chart.rows):
            strip = np.full((h, target_w, 3), 60, dtype=np.uint8)
            cv2.putText(strip, "(yok)", (target_w // 2 - 30, h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return strip

        row_palette_idx = chart.grid[row_idx]
        # her hücre cell_px genişliğinde
        per_cell_w = max(1, target_w // chart.cols)
        used_w = per_cell_w * chart.cols
        strip = np.zeros((h, used_w, 3), dtype=np.uint8)
        for c, p_idx in enumerate(row_palette_idx):
            color_bgr = chart.palette_bgr[p_idx]
            strip[:, c * per_cell_w:(c + 1) * per_cell_w] = color_bgr
        for c in range(chart.cols + 1):
            x = c * per_cell_w
            cv2.line(strip, (x, 0), (x, h), (40, 40, 40), 1)

        if used_w < target_w:
            pad = np.full((h, target_w - used_w, 3), 30, dtype=np.uint8)
            strip = np.hstack([strip, pad])
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
        panel = np.full((height, self.panel_width, 3), 30, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30

        cv2.putText(panel, "MOTIFIKA", (12, y), font, 0.9, (50, 220, 220), 2)
        y += 30
        cv2.putText(panel, f"Sira: {active_row} / {chart.rows}",
                    (12, y), font, 0.7, (255, 255, 255), 2)
        y += 25
        progress = (
            (chart.rows - active_row) / chart.rows
            if direction == "bottom_up"
            else active_row / chart.rows
        )
        bar_w = self.panel_width - 24
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 14), (60, 60, 60), -1)
        cv2.rectangle(panel, (12, y), (12 + int(bar_w * progress), y + 14),
                      (50, 200, 50), -1)
        y += 30

        cv2.putText(panel, "Aktif sira", (12, y), font, 0.6, (180, 220, 255), 1)
        y += 8
        active_strip = self.render_next_strip(chart, active_row, 0, direction, self.panel_width - 24)
        panel[y:y + active_strip.shape[0], 12:12 + active_strip.shape[1]] = active_strip
        y += active_strip.shape[0] + 12

        cv2.putText(panel, "Sonraki sira", (12, y), font, 0.6, (255, 220, 180), 1)
        y += 8
        next1 = self.render_next_strip(chart, active_row, 1, direction, self.panel_width - 24)
        panel[y:y + next1.shape[0], 12:12 + next1.shape[1]] = next1
        y += next1.shape[0] + 8

        cv2.putText(panel, "Sonra", (12, y), font, 0.5, (200, 200, 200), 1)
        y += 6
        next2 = self.render_next_strip(chart, active_row, 2, direction, self.panel_width - 24)
        panel[y:y + next2.shape[0], 12:12 + next2.shape[1]] = next2
        y += next2.shape[0] + 16

        # Sıra renk özeti (Tetris next-piece açıklaması)
        next_row_idx = (
            active_row - 1 if direction == "bottom_up" else active_row + 1
        )
        if 0 <= next_row_idx < chart.rows:
            counts: dict[int, int] = {}
            for p_idx in chart.grid[next_row_idx]:
                counts[int(p_idx)] = counts.get(int(p_idx), 0) + 1
            parts = []
            for p_idx, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                name = _color_name(chart.palette_rgb[p_idx])
                parts.append(f"{n} {name}")
            cv2.putText(panel, "Sonraki: " + ", ".join(parts[:3]),
                        (12, y), font, 0.55, (180, 220, 255), 1)
            y += 24

        # Uyarılar
        if mismatches:
            cv2.putText(panel, f"UYARI ({len(mismatches)} hata)",
                        (12, y), font, 0.7, (60, 60, 240), 2)
            y += 26
            row_label = check_row if check_row is not None else active_row
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

        # Klavye yardımı
        y_help = height - 50
        for line in ["[+/-] sira ayarla", "[r] kalibrasyon", "[q] cikis"]:
            cv2.putText(panel, line, (12, y_help), font, 0.5, (160, 160, 160), 1)
            y_help += 18
        return panel

    def compose(self, camera_view: np.ndarray, panel: np.ndarray) -> np.ndarray:
        h = camera_view.shape[0]
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (panel.shape[1], h))
        return np.hstack([camera_view, panel])
