"""El çizimi (kareli kağıt) motiflerden temiz chart üret.

Kullanıcı önerisi: yüksek çözünürlükte pikselleştir, sonra çok sayıda kaynak
pikselini tek hücreye topla. Burada her kaynak piksel önce kırmızı/arka-plan
olarak SINIFLANDIRILIR (full çözünürlük), sonra her chart hücresi için ÇOĞUNLUK
oyu alınır. Kareli kağıdın gri çizgileri "kırmızı" sayılmadığından hücreleri
kirletmez → motif net ve simetrik çıkar.

Akış: kırmızı maske → kareli kağıt ızgarasını tespit edip kırp → en-boy oranını
koruyarak hedef ızgaraya çoğunlukla küçült → arka planla R×C'ye doldur (ortala).

Kullanım:
    python -m tools.regen_motif_charts            # 44×38 (varsayılan), tüm motifler
    python -m tools.regen_motif_charts --rows 44 --cols 38
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.pattern import save_chart

# motif anahtarı → kaynak görsel (kareli kağıt fotoğrafı).
SOURCES = {
    "eli_belinde": Path("eli_belinde.jpeg"),
    "hayat_agaci": Path("hayat_agaci.jpeg"),
}


def _red_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Kırmızı motif pikselleri → float maske (1.0 kırmızı, 0.0 değil)."""
    r, g, b = (img_bgr[:, :, 2].astype(int),
               img_bgr[:, :, 1].astype(int),
               img_bgr[:, :, 0].astype(int))
    return ((r > 110) & (r - g > 40) & (r - b > 40)).astype(np.float32)


def _line_positions(prof: np.ndarray, min_dist: int = 40, frac: float = 0.30) -> list[int]:
    """1B koyuluk profilindeki ızgara çizgisi tepe konumları (min_dist aralıklı)."""
    prof = np.convolve(prof, np.ones(5) / 5, "same")
    thr = prof.min() + (prof.max() - prof.min()) * frac
    pos: list[int] = []
    last = -1e9
    for i in range(1, len(prof) - 1):
        if prof[i] >= thr and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and i - last >= min_dist:
            pos.append(i)
            last = i
    return pos


def build_motif_chart(src: Path, rows: int, cols: int) -> dict:
    """Kareli kağıt motifinden rows×cols chart sözlüğü (2 renk, çoğunluk oyu)."""
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {src}")
    mask = _red_mask(img)
    ys, xs = np.where(mask > 0)
    y0, x0 = ys.min(), xs.min()

    # Motifin üstündeki / solundaki kenar boşluğunda yalnız ızgara çizgileri var;
    # oradan tüm yatay/dikey çizgi konumlarını çıkarıp kareli alana kırpıyoruz.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    xp = _line_positions(255 - gray[5:max(20, y0 - 8), :].mean(0))
    yp = _line_positions(255 - gray[:, 5:max(20, x0 - 8)].mean(1))
    cm = mask[yp[0]:yp[-1], xp[0]:xp[-1]]
    g_rows, g_cols = len(yp) - 1, len(xp) - 1

    # En-boy oranını koru: motifi R×C'ye SIĞDIR (gerekirse arka planla doldur).
    scale = min(rows / g_rows, cols / g_cols)
    nr, nc = min(rows, round(g_rows * scale)), min(cols, round(g_cols * scale))

    # INTER_AREA ikili maskede = her hücredeki kırmızı oranı → 0.5 üstü çoğunluk.
    frac = cv2.resize(cm, (nc, nr), interpolation=cv2.INTER_AREA)
    motif = (frac > 0.5).astype(int)

    # Palet renkleri kaynaktan (kamera renk eşleşmesi için temsilî): 0=arka plan, 1=kırmızı.
    red_rgb = img[mask > 0][:, ::-1].mean(0)
    bg_rgb = img[mask == 0][:, ::-1].mean(0)
    palette = np.array([bg_rgb, red_rgb]).round().astype(int)

    grid = np.zeros((rows, cols), int)
    top, left = (rows - nr) // 2, (cols - nc) // 2
    grid[top:top + nr, left:left + nc] = motif

    return {
        "source": src.name,
        "rows": rows,
        "cols": cols,
        "palette": palette.tolist(),
        "grid": grid.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=44)
    ap.add_argument("--cols", type=int, default=38)
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    args = ap.parse_args()

    for key, src in SOURCES.items():
        chart = build_motif_chart(src, args.rows, args.cols)
        out = args.assets / f"{key}_chart.json"
        save_chart(chart, out)
        red = chart["palette"][1]
        print(f"{key}: {args.rows}×{args.cols} kaydedildi → {out} (kırmızı RGB={red})")


if __name__ == "__main__":
    main()
