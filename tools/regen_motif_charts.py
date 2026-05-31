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


def _best_grid(profile: np.ndarray, lo: float = 70.0, hi: float = 80.0,
               step: float = 0.2) -> tuple[float, int]:
    """1B koyuluk profilinde en iyi UNIFORM ızgarayı (periyot, faz) bul.

    Kareli kağıt elle çizilip hafif perspektif/eğimle çekildiğinden tek tek çizgi
    tespiti gürültülü (çizgi kaçar/uydurulur → hücreler kayar → artifact). Bunun
    yerine tüm görüntü boyunca periyot ve fazı tarayıp, periyodik 'tarak'
    konumlarındaki ORTALAMA koyuluğu maksimum yapan ızgarayı seçeriz: tek bir
    çizgiye değil, bütün ızgaraya küresel olarak oturur → kayma/artifact yok.
    Periyot aralığı (lo..hi) bu kaynak fotoğraflarının hücre boyutuna (~74-78px) göre.
    """
    n = len(profile)
    best = None
    p = lo
    while p <= hi:
        for o in range(int(round(p))):
            pos = np.round(np.arange(o, n - 1, p)).astype(int)
            if len(pos) >= 10:
                s = float(profile[pos].mean())
                if best is None or s > best[0]:
                    best = (s, p, o)
        p += step
    return best[1], best[2]  # (periyot, faz)


def _grid_lines(period: float, phase: int, n: int) -> np.ndarray:
    """phase'ten başlayıp period aralıkla n'e kadar ızgara çizgisi konumları."""
    L = np.round(np.arange(phase, n - 1, period)).astype(int)
    return L[L >= 0]


def build_motif_chart(src: Path, rows: int | None = None, cols: int | None = None) -> dict:
    """Kareli kağıt motifinden chart sözlüğü (2 renk, hücre başına çoğunluk oyu).

    Izgara, tüm-görüntü periyot+faz EN-İYİ-UYUMUYLA bulunur (_best_grid; perspektife
    dayanıklı, kayma yok), motifin kırmızı bbox'ına +marj kırpılır ve her hücre
    çoğunluk oyuyla örneklenir. Her kağıt karesi = bir chart hücresi = bir dokuma
    düğümü → PİKSEL KAYBI / artifact yok, simetri korunur. rows/cols VERİLMEZSE bu
    gerçek ızgara kullanılır; verilirse o boyuta yeniden örneklenir (override; gerçek
    ızgaradan farklıysa piksel kaybedebilir).
    """
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {src}")
    mask = _red_mask(img)
    dark = 255.0 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    ys, xs = np.where(mask > 0)

    # Satır ızgarası tüm GENİŞLİK, sütun ızgarası tüm YÜKSEKLİK profilinden (çok veri
    # → sağlam). Çizgiler tüm görüntü boyunca düzgün uniform kabul edilir.
    pr, orow = _best_grid(dark.mean(1))
    pc, ocol = _best_grid(dark.mean(0))
    RL = _grid_lines(pr, orow, img.shape[0])
    CL = _grid_lines(pc, ocol, img.shape[1])

    # Motif kırmızı bbox'ını ızgara çizgilerine indeksle (+MARGIN hücre çerçeve marjı).
    MARGIN = 2
    ri0 = max(0, int(np.searchsorted(RL, ys.min())) - MARGIN)
    ri1 = min(len(RL) - 1, int(np.searchsorted(RL, ys.max())) + MARGIN)
    ci0 = max(0, int(np.searchsorted(CL, xs.min())) - MARGIN)
    ci1 = min(len(CL) - 1, int(np.searchsorted(CL, xs.max())) + MARGIN)
    RLc, CLc = RL[ri0:ri1 + 1], CL[ci0:ci1 + 1]
    g_rows, g_cols = len(RLc) - 1, len(CLc) - 1

    # Her hücre bir kağıt karesi → içindeki kırmızı oranı 0.5 üstü ise dolu (çoğunluk).
    motif = np.zeros((g_rows, g_cols), int)
    for i in range(g_rows):
        for j in range(g_cols):
            if mask[RLc[i]:RLc[i + 1], CLc[j]:CLc[j + 1]].mean() > 0.5:
                motif[i, j] = 1

    # Palet renkleri kaynaktan (kamera renk eşleşmesi için temsilî): 0=arka plan, 1=kırmızı.
    red_rgb = img[mask > 0][:, ::-1].mean(0)
    bg_rgb = img[mask == 0][:, ::-1].mean(0)
    palette = np.array([bg_rgb, red_rgb]).round().astype(int)

    if rows is None or cols is None:
        # GERÇEK ızgara — yeniden örnekleme yok (motif net, simetrik, piksel kaybı yok).
        return {
            "source": src.name, "rows": g_rows, "cols": g_cols,
            "palette": palette.tolist(), "grid": motif.tolist(),
        }

    # Override: en-boy oranını koru, R×C'ye SIĞDIR (arka planla ortala; lossy olabilir).
    scale = min(rows / g_rows, cols / g_cols)
    nr, nc = min(rows, max(1, round(g_rows * scale))), min(cols, max(1, round(g_cols * scale)))
    small = (cv2.resize(motif.astype(np.float32), (nc, nr), interpolation=cv2.INTER_AREA) > 0.5).astype(int)
    grid = np.zeros((rows, cols), int)
    top, left = (rows - nr) // 2, (cols - nc) // 2
    grid[top:top + nr, left:left + nc] = small
    return {
        "source": src.name, "rows": rows, "cols": cols,
        "palette": palette.tolist(), "grid": grid.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # Varsayılan: rows/cols VERME → motifin tespit edilen GERÇEK ızgarası kullanılır
    # (piksel kaybı/artifact yok). Override istersen --rows/--cols ver.
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    args = ap.parse_args()

    for key, src in SOURCES.items():
        chart = build_motif_chart(src, args.rows, args.cols)
        out = args.assets / f"{key}_chart.json"
        save_chart(chart, out)
        red = chart["palette"][1]
        print(f"{key}: {chart['rows']}×{chart['cols']} (gerçek ızgara) kaydedildi → {out} (kırmızı RGB={red})")


if __name__ == "__main__":
    main()
