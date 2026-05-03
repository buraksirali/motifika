"""Teknik validasyon: bozuk dosya, çok küçük, yanlış format ele."""
from pathlib import Path
import cv2
import sys

def validate_folder(folder: str, min_size: int = 300):
    folder_path = Path(folder)
    files = sorted(folder_path.iterdir())
    bad = []
    ok = []

    for p in files:
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            bad.append((p, f"format_{p.suffix}"))
            continue
        img = cv2.imread(str(p))
        if img is None:
            bad.append((p, "okunamadi"))
            continue
        h, w = img.shape[:2]
        if min(h, w) < min_size:
            bad.append((p, f"kucuk_{min(h,w)}px"))
            continue
        ok.append((p, w, h))

    print(f"\n=== {folder} ===")
    print(f"OK: {len(ok)}  BAD: {len(bad)}")
    for p, reason in bad:
        print(f"  SİL [{reason}] {p.name}")
        p.unlink()
    return len(ok)

if __name__ == "__main__":
    for f in sys.argv[1:]:
        validate_folder(f)
