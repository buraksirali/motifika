"""Ekran düzeni: SOL = kamera + AR overlay, SAĞ = next-piece paneli + Türkçe uyarı.

Sağ panel (Tetris next-piece tarzı): başlık, sıra sayacı, ilerleme çubuğu,
önceki/şimdiki/sonraki sıra şeritleri, sıradaki renk özeti, hata uyarıları,
klavye yardımı. compose() kamera görüntüsünü panelle yan yana birleştirir.

Metin çizimi PIL ile yapılır: cv2.putText Hershey fontları Türkçe karakterleri
(ı, ş, ğ, ç, ö, ü) "?" gösteriyordu; PIL + DejaVu TTF doğru render eder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_WIDTH = 480
PANEL_MARGIN = 24             # panel iç kenar boşluğu (sol/sağ/üst)
NEXT_ROW_CELL_PX = 26         # sağdaki şerit hücresi (overlay'den büyük, göz çeksin)

# Türkçe karakter içeren DejaVu TTF — Hershey'in aksine ı/ş/ğ/ç/ö/ü destekler.
# Donmuş (PyInstaller) Pi'de sistem fontu olmayabilir → önce sürümle gelen fonts/
# klasörüne, sonra repo köküne, en son sisteme bakarız.
_FONT_REGULAR_NAME = "DejaVuSans.ttf"
_FONT_BOLD_NAME = "DejaVuSans-Bold.ttf"
_font_cache: dict[tuple[int, bool], "ImageFont.FreeTypeFont"] = {}


def _find_font(filename: str) -> str:
    """DejaVu TTF'i sırayla ara: data-root/fonts → repo_kökü/fonts → sistem."""
    candidates = [
        Path("fonts") / filename,                                       # data-root/fonts (frozen chdir'lı veya cwd)
        Path(__file__).resolve().parent.parent / "fonts" / filename,    # repo_kökü/fonts (geliştirme)
        Path("/usr/share/fonts/truetype/dejavu") / filename,            # sistem (apt fonts-dejavu-core)
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[-1])  # son çare: sistem yolu (yoksa _get_font load_default'a düşer)

# Şu an KULLANILMIYOR; ileride kullanıcı paletinde Türkçe etiket için.
TURKISH_COLOR_NAMES = {
    "kirmizi": "Kırmızı", "lacivert": "Lacivert", "krem": "Krem",
    "siyah": "Siyah", "beyaz": "Beyaz", "kahverengi": "Kahverengi",
    "yesil": "Yeşil", "sari": "Sarı", "mavi": "Mavi", "bordo": "Bordo",
}


def _get_font(px: int, bold: bool) -> "ImageFont.FreeTypeFont":
    """İstenen boyut/kalınlıkta TTF font (cache'li — her karede yeniden açma)."""
    key = (px, bold)
    font = _font_cache.get(key)
    if font is None:
        try:
            font = ImageFont.truetype(
                _find_font(_FONT_BOLD_NAME if bold else _FONT_REGULAR_NAME), px)
        except OSError:
            font = ImageFont.load_default()  # TTF yoksa son çare
        _font_cache[key] = font
    return font


def _draw_texts(img_bgr: np.ndarray, items: list) -> None:
    """Bir dizi metni tek PIL geçişiyle img üstüne çiz (UTF-8 / Türkçe güvenli).

    items: [(metin, (x, y), px, renk_bgr, bold), ...]; y = metnin TABAN çizgisi.
    Tek BGR↔RGB dönüşümü: her metin için ayrı dönüşüm yapmaktan çok daha hızlı.
    """
    if not items:
        return
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, org, px, color_bgr, bold in items:
        font = _get_font(px, bold)
        rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        # anchor="ls": x sola, y taban çizgisine hizalı (cv2.putText org'una yakın).
        draw.text(org, text, font=font, fill=rgb, anchor="ls")
    img_bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


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
    margin: int = PANEL_MARGIN
    next_row_cell_px: int = NEXT_ROW_CELL_PX

    def render_next_strip(
        self,
        chart,
        active_row: int,
        offset: int,
        target_w: int,
    ) -> np.ndarray:
        """Aktif sıradan offset kadar uzaktaki sıranın renk şeridi (tam target_w piksel).

        offset geometrik: -1 chart'ta bir üst sıra, 0 şimdiki, +1 bir alt sıra.
        Şeridin "önceki/sonraki" anlamı dokuma yönüne göre render_panel'de verilir.
        """
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
        m = self.margin
        content_w = self.panel_width - 2 * m
        texts: list = []  # (metin, org, px, renk_bgr, bold) — sonda tek geçişte çizilir
        y = 46  # aşağı kayan metin taban çizgisi

        # Başlık.
        texts.append(("MOTİFİKA", (m, y), 30, (50, 220, 220), True))
        y += 42

        # Sıra sayacı.
        texts.append((f"Sıra: {active_row} / {chart.rows}",
                       (m, y), 22, (255, 255, 255), True))
        y += 18

        # İlerleme oranı: bottom_up'ta active küçüldükçe artar, top_down'da tersi.
        progress = (
            (chart.rows - active_row) / chart.rows
            if direction == "bottom_up"
            else active_row / chart.rows
        )

        # İlerleme çubuğu (arka plan + yeşil dolgu, kalınlık -1 = dolu).
        cv2.rectangle(panel, (m, y), (m + content_w, y + 18), (60, 60, 60), -1)
        cv2.rectangle(panel, (m, y), (m + int(content_w * progress), y + 18),
                      (50, 200, 50), -1)
        y += 18 + 40

        # Önceki / şimdiki / sonraki sıra şeritleri.
        # Şeritler GEOMETRİK sırada dizilir: üstte küçük indeks (offset -1),
        # ortada şimdiki, altta büyük indeks (offset +1) — chart yukarıdan aşağı.
        # Etiketler dokuma yönüne göre değişir: top_down'da yukarı çıkan dokumada
        # üstteki sıra "önceki" (bitmiş); bottom_up'ta üstteki sıra "sonraki".
        prev_lbl = ("Önceki sıra", (160, 160, 160), False)
        curr_lbl = ("Şimdiki sıra", (50, 220, 255), True)
        next_lbl = ("Sonraki sıra", (255, 220, 180), False)
        if direction == "bottom_up":
            top_lbl, bot_lbl = next_lbl, prev_lbl
        else:
            top_lbl, bot_lbl = prev_lbl, next_lbl

        ordered = [(*top_lbl, -1), (*curr_lbl, 0), (*bot_lbl, +1)]
        for label, color, bold, offset in ordered:
            texts.append((label, (m, y), 19, color, bold))
            strip_top = y + 12
            strip = self.render_next_strip(chart, active_row, offset, content_w)
            # Slicing assign: paneldeki bölgeyi şeritle override et.
            panel[strip_top:strip_top + strip.shape[0], m:m + strip.shape[1]] = strip
            y = strip_top + strip.shape[0] + 38

        # Sonraki (yapılacak) sıranın renk özeti ("3 Kırmızı, 5 Siyah").
        # bottom_up'ta sonraki sıra yukarıda (active-1), top_down'da aşağıda (active+1).
        next_row_idx = active_row - 1 if direction == "bottom_up" else active_row + 1
        if 0 <= next_row_idx < chart.rows:
            counts: dict[int, int] = {}
            for p_idx in chart.grid[next_row_idx]:
                counts[int(p_idx)] = counts.get(int(p_idx), 0) + 1

            parts = []
            # Adet çok olan renk önce (key=-değer ile azalan sıralama).
            for p_idx, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                parts.append(f"{n} {_color_name(chart.palette_rgb[p_idx])}")
            # İlk 3 renk (paneli taşırmasın).
            texts.append(("Sonraki: " + ", ".join(parts[:3]),
                          (m, y), 16, (180, 220, 255), False))
            y += 38

        # Renk uyarıları.
        if mismatches:
            texts.append((f"UYARI ({len(mismatches)} hata)",
                          (m, y), 22, (60, 60, 240), True))
            y += 32
            # Kontrol aktif sırada değil bir önceki tamamlanmış sırada yapılıyor.
            row_label = check_row if check_row is not None else active_row
            # En çok 5 hata göster (ekrana sığsın).
            for col, exp_idx, obs_idx, dist in mismatches[:5]:
                exp = _color_name(chart.palette_rgb[exp_idx])
                obs = _color_name(chart.palette_rgb[obs_idx])
                line = f"S{row_label}.{col}: {exp} yerine {obs}"
                texts.append((line, (m, y), 16, (120, 200, 255), False))
                y += 24
        else:
            texts.append(("Renk uyumu: OK", (m, y), 19, (120, 220, 120), True))
            y += 24

        # Klavye yardımı — panelin en altına sabit.
        y_help = height - 100
        for line in ["[yukarı/aşağı ok] sıra", "[z/x] yakınlaş/uzaklaş",
                     "[r] kalibrasyon", "[d] yön değiştir", "[q] çıkış"]:
            texts.append((line, (m, y_help), 15, (160, 160, 160), False))
            y_help += 22

        # Tüm metinleri tek PIL geçişiyle çiz (Türkçe karakterler doğru görünsün).
        _draw_texts(panel, texts)
        return panel

    def compose(self, camera_view: np.ndarray, panel: np.ndarray,
                vertical: bool = False) -> np.ndarray:
        """Kamera görüntüsü + paneli birleştir.

        vertical=False (varsayılan, yatay): kamera solda, panel sağda (yükseklikleri eşitlenir).
        vertical=True (portrait): kamera üstte, panel altta (genişlikleri eşitlenir).
        """
        if vertical:
            w = camera_view.shape[1]
            # Panel genişliği kameraya uymuyorsa uydur.
            if panel.shape[1] != w:
                panel = cv2.resize(panel, (w, panel.shape[0]))
            return np.vstack([camera_view, panel])

        h = camera_view.shape[0]
        # Panel yüksekliği kameraya uymuyorsa uydur.
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (panel.shape[1], h))
        return np.hstack([camera_view, panel])
