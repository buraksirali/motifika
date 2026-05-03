"""
Görselleri eğitime uygunluk açısından sınıflandırır.

Kategoriler:
  KEEP   — gerçek dokunmuş kilim/halı fotoğrafı (eğitim verisi)
  REJECT — vektör tasarım, şematik çizim, web ekran görüntüsü, alakasız nesne

Heuristikler (etiketlenmemiş veride çalışır):
  1. En boy oranı uç değerleri (banner/screenshot ele) → REJECT
  2. Min boyut < 400px → REJECT (Pi Camera 640+ ile çalışır)
  3. Renk paleti çeşitliliği (k=8 kümeleme entropisi) → düşükse vektör
  4. Doku entropisi (gri tonlamalı GLCM benzeri) → yüksekse gerçek dokuma
  5. Kenar yoğunluğu + dağılımı → vektörler keskin, kilimler yumuşak/dağınık
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from skimage.measure import shannon_entropy

ROOT = Path("/home/burakeda/Projeler/motifika")
FOLDERS = ["hayat agacı", "motif eli belinde"]


def score_image(path: Path) -> tuple[bool, str, dict]:
    img = cv2.imread(str(path))
    if img is None:
        return False, "okunamadi", {}

    h, w = img.shape[:2]
    ar = w / h
    metrics = {"w": w, "h": h, "ar": round(ar, 2)}

    if min(w, h) < 400:
        return False, f"kucuk_{min(w,h)}px", metrics
    if ar < 0.5 or ar > 2.2:
        return False, f"asiri_oran_{ar:.2f}", metrics

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        small = img

    tex_entropy = float(shannon_entropy(gray))
    metrics["entropy"] = round(tex_entropy, 2)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean()) / 255.0
    metrics["edge_density"] = round(edge_density, 3)

    pixels = small.reshape(-1, 3).astype(np.float32)
    sample = pixels[np.random.choice(len(pixels), size=min(5000, len(pixels)), replace=False)]
    _, labels, _ = cv2.kmeans(
        sample, 8, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=8) / len(labels)
    palette_entropy = float(-(counts * np.log2(counts + 1e-9)).sum())
    metrics["palette_entropy"] = round(palette_entropy, 2)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics["blur"] = round(blur, 1)

    if tex_entropy < 5.5:
        return False, f"dusuk_doku_entropi_{tex_entropy:.1f}", metrics
    if palette_entropy < 1.8:
        return False, f"basit_palet_{palette_entropy:.1f}", metrics
    if edge_density > 0.18:
        return False, f"asiri_kenar_{edge_density:.3f}", metrics
    if edge_density < 0.015:
        return False, f"dusuk_kenar_{edge_density:.3f}", metrics
    if blur < 80:
        return False, f"bulanik_{blur:.0f}", metrics

    return True, "ok", metrics


def main():
    np.random.seed(0)
    total_keep = 0
    total_reject = 0
    report_lines = []

    for folder in FOLDERS:
        folder_path = ROOT / folder
        files = sorted([p for p in folder_path.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        keep_dir = folder_path / "_keep"
        reject_dir = folder_path / "_reject"
        keep_dir.mkdir(exist_ok=True)
        reject_dir.mkdir(exist_ok=True)

        report_lines.append(f"\n=== {folder} ({len(files)} dosya) ===")
        kept = rejected = 0

        for p in files:
            ok, reason, m = score_image(p)
            if ok:
                target = keep_dir / p.name
                kept += 1
            else:
                target = reject_dir / p.name
                rejected += 1
                report_lines.append(f"  REJECT [{reason}] {m} {p.name[:60]}")
            try:
                p.rename(target)
            except OSError:
                pass

        report_lines.append(f"  >>> KEEP: {kept}  REJECT: {rejected}")
        total_keep += kept
        total_reject += rejected

    print("\n".join(report_lines))
    print(f"\n=== GENEL TOPLAM ===\nKEEP: {total_keep}\nREJECT: {total_reject}")


if __name__ == "__main__":
    main()
