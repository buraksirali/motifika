"""Motif görselini NxM ızgaralı chart'a çevirir.

Kullanım:
    python -m app.pattern eli_belinde.jpg --rows 40 --cols 40 --palette 4 \
        --out assets/eli_belinde_chart.json

Çıktı JSON şeması:
    {
        "source": "eli_belinde.jpg",
        "rows": 40, "cols": 40,
        "palette": [[r,g,b], ...],   # 0..255 BGR sırasında değil, RGB
        "grid": [[palette_idx, ...], ...]   # rows x cols
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def flatten_alpha(img: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    """PNG alpha kanalını beyaz zemine düzleştir."""
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
        out = bgr * alpha + bg_arr * (1.0 - alpha)
        return out.astype(np.uint8)
    return img


def pixelate(img_bgr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Görseli rows × cols ızgaraya küçült (her hücre = bir piksel)."""
    return cv2.resize(img_bgr, (cols, rows), interpolation=cv2.INTER_AREA)


def quantize_palette(small_bgr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means ile k renkli palet bul. Dönüş: (palette_rgb [k,3], grid [rows,cols] int)."""
    rows, cols = small_bgr.shape[:2]
    pixels = small_bgr.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )

    palette_bgr = np.clip(centers, 0, 255).astype(np.uint8)
    palette_rgb = palette_bgr[:, ::-1]
    grid = labels.reshape(rows, cols).astype(int)
    return palette_rgb, grid


def render_preview(palette_rgb: np.ndarray, grid: np.ndarray, cell_px: int = 16) -> np.ndarray:
    """Chart'ı görsel olarak doğrulamak için BGR önizleme oluştur."""
    palette_bgr = palette_rgb[:, ::-1]
    preview = palette_bgr[grid]
    preview = np.repeat(np.repeat(preview, cell_px, axis=0), cell_px, axis=1)
    h, w = preview.shape[:2]
    for r in range(grid.shape[0] + 1):
        cv2.line(preview, (0, r * cell_px), (w, r * cell_px), (40, 40, 40), 1)
    for c in range(grid.shape[1] + 1):
        cv2.line(preview, (c * cell_px, 0), (c * cell_px, h), (40, 40, 40), 1)
    return preview


def build_chart(image_path: Path, rows: int, cols: int, palette_size: int) -> dict:
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Görsel okunamadı: {image_path}")
    bgr = flatten_alpha(raw)
    small = pixelate(bgr, rows, cols)
    palette_rgb, grid = quantize_palette(small, palette_size)

    return {
        "source": image_path.name,
        "rows": rows,
        "cols": cols,
        "palette": palette_rgb.tolist(),
        "grid": grid.tolist(),
    }


def save_chart(chart: dict, out_path: Path, preview: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chart, ensure_ascii=False, indent=2))
    if preview:
        palette_rgb = np.array(chart["palette"], dtype=np.uint8)
        grid = np.array(chart["grid"], dtype=int)
        prev = render_preview(palette_rgb, grid)
        prev_path = out_path.with_suffix(".preview.png")
        cv2.imwrite(str(prev_path), prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="kaynak motif görseli")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--palette", type=int, default=4, help="renk sayısı (k-means)")
    ap.add_argument("--out", type=Path, required=True, help="çıktı chart.json yolu")
    args = ap.parse_args()

    np.random.seed(0)
    chart = build_chart(args.image, args.rows, args.cols, args.palette)
    save_chart(chart, args.out)
    print(f"chart kaydedildi: {args.out}")
    print(f"önizleme: {args.out.with_suffix('.preview.png')}")
    print(f"palet: {chart['palette']}")


if __name__ == "__main__":
    main()
