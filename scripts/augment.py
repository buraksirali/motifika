"""
Seçilen kilim görsellerini Pi Camera deploy ortamına benzeyen varyantlara çoğaltır.

Her kaynak görselden 12 augmente kopya üretilir:
  - 3 perspektif/rotasyon (kamera açısı varyasyonu)
  - 3 ışık/parlaklık (atölye ışığı, gölge, gündüz)
  - 2 motion blur (kullanıcı eli titremesi)
  - 2 ISO noise (Pi Camera düşük ışık gürültüsü)
  - 1 renk değişimi (HSV shift, hatalı renk benzetimi için seed)
  - 1 crop + zoom (kompozisyon farklılığı)

Çıktı: <kaynak_klasör>/_augmented/
Hedef: 65 görsel × 12 = ~780 augmente görsel (toplam ~845 görsel)
"""

from pathlib import Path
import cv2
import numpy as np
import albumentations as A

ROOT = Path("/home/burakeda/Projeler/motifika")
FOLDERS = ["hayat agacı", "motif eli belinde"]
TARGET_SIZE = 640


def build_pipelines():
    return [
        ("perspektif1", A.Compose([
            A.Perspective(scale=(0.05, 0.10), p=1.0),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ])),
        ("perspektif2", A.Compose([
            A.Affine(scale=(0.85, 1.15), shear=(-10, 10), translate_percent=(-0.05, 0.05), p=1.0),
        ])),
        ("perspektif3", A.Compose([
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.HorizontalFlip(p=0.5),
        ])),
        ("isik_parlak", A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0),
        ])),
        ("isik_golge", A.Compose([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), contrast_limit=0.1, p=1.0),
        ])),
        ("isik_gunduz", A.Compose([
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=2.0, p=0.7),
        ])),
        ("blur_motion1", A.Compose([
            A.MotionBlur(blur_limit=(5, 11), p=1.0),
        ])),
        ("blur_motion2", A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ])),
        ("noise_iso", A.Compose([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.2, 0.5), p=1.0),
        ])),
        ("noise_gauss", A.Compose([
            A.GaussNoise(var_limit=(20.0, 60.0), p=1.0),
        ])),
        ("renk_hsv", A.Compose([
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
        ])),
        ("crop_zoom", A.Compose([
            A.RandomResizedCrop(size=(TARGET_SIZE, TARGET_SIZE), scale=(0.5, 0.9), ratio=(0.85, 1.15), p=1.0),
        ])),
    ]


def fit_to_target(img: np.ndarray) -> np.ndarray:
    """Letterbox to TARGET_SIZE x TARGET_SIZE preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), 114, dtype=np.uint8)
    y0, x0 = (TARGET_SIZE - nh) // 2, (TARGET_SIZE - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main():
    np.random.seed(42)
    pipelines = build_pipelines()
    grand_total = 0

    for folder in FOLDERS:
        keep_dir = ROOT / folder / "_keep"
        out_dir = ROOT / folder / "_augmented"
        out_dir.mkdir(exist_ok=True)

        files = sorted([p for p in keep_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

        produced = 0
        for src in files:
            img = cv2.imread(str(src))
            if img is None:
                continue
            base = fit_to_target(img)
            stem = src.stem[:40].replace(" ", "_").replace("/", "_")

            orig_out = out_dir / f"{stem}__orig.jpg"
            cv2.imwrite(str(orig_out), base, [cv2.IMWRITE_JPEG_QUALITY, 92])
            produced += 1

            for tag, pipe in pipelines:
                try:
                    aug = pipe(image=base)["image"]
                    if aug.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
                        aug = cv2.resize(aug, (TARGET_SIZE, TARGET_SIZE))
                    out_path = out_dir / f"{stem}__{tag}.jpg"
                    cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    produced += 1
                except Exception as e:
                    print(f"  HATA {src.name} / {tag}: {e}")

        print(f"{folder}: {len(files)} kaynak → {produced} augmente görsel")
        grand_total += produced

    print(f"\n=== TOPLAM AUGMENTE GÖRSEL: {grand_total} ===")


if __name__ == "__main__":
    main()
