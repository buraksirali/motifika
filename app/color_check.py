"""Bir sıradaki hücre renklerini çıkar ve beklenen palette ile karşılaştır.

Tabloda kırmızı olması gereken yerde siyah dokunduysa "S5.12: Kırmızı yerine
Siyah" diye uyarır.

İki backend:
  - HSVBackend (varsayılan): klasik OpenCV, eğitim gerektirmez. Demo bunu kullanır.
  - HailoBackend (placeholder): Hailo-8L .hef inference. H5'te eklenecek.

API:
    backend = HSVBackend(palette_rgb)
    mismatches = check_active_row(frame, H_cam_to_chart, chart, active_row, backend)
    # mismatches: [(col, expected_idx, observed_idx, distance), ...]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


CELL_PX = 12

# CIE-LAB'da ~25 birim = göze çarpan fark. Şu an KULLANILMIYOR;
# kontrol sadece indeks farkına bakıyor. İleride eşik filtresi için.
LAB_DISTANCE_THRESHOLD = 25.0


class ColorBackend(Protocol):
    """Renk sınıflandırıcı arabirimi. Yeni backend bu metoda sahip olmalı."""
    def classify(self, samples_bgr: np.ndarray) -> np.ndarray:
        """N×3 örnek için palet indeksi (N,) döndür."""
        ...


@dataclass
class HSVBackend:
    """LAB uzayında en yakın palet rengini bul (klasik, hızlı, eğitim gerektirmez).

    NOT: Adı HSV ama LAB kullanıyor (HSV erken denemelerden kaldı, isim değişikliği
    kodu kırmamak için ertelendi).
    """
    palette_rgb: np.ndarray  # (k, 3) uint8

    def __post_init__(self):
        # Paleti LAB'a önişle. cvtColor "resim" ister → [None,:,:] ile (1,k,3) yap.
        # LAB seçildi çünkü LAB'da Öklid mesafesi ≈ göze görünen renk farkı.
        bgr = self.palette_rgb[:, ::-1].astype(np.uint8)[None, :, :]
        self._palette_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0].astype(np.float32)

    def classify(self, samples_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """N×3 örnek için (en yakın palet indeksi, LAB mesafesi) döndür."""
        if samples_bgr.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        # Örnekleri LAB'a çevir (cvtColor için (N,1,3), sonra (N,3)).
        lab = cv2.cvtColor(samples_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        lab = lab.reshape(-1, 3).astype(np.float32)

        # Broadcasting ile (N,k) mesafe matrisi: (N,1,3) - (1,k,3) → (N,k,3).
        diff = lab[:, None, :] - self._palette_lab[None, :, :]
        dist = np.linalg.norm(diff, axis=2)  # son eksen → (N,k) Öklid

        idx = dist.argmin(axis=1)    # her örnek için en yakın palet indeksi
        min_dist = dist.min(axis=1)
        return idx, min_dist


@dataclass
class HailoBackend:
    """Hailo-8L .hef wrapper. H5'te implementasyon yapılacak; şimdilik HSV'ye düşer."""
    palette_rgb: np.ndarray
    hef_path: str | None = None

    def __post_init__(self):
        # Hailo kullanılamazsa HSV yedek (graceful degradation).
        self._fallback = HSVBackend(self.palette_rgb)
        self._available = False
        try:
            # PLACEHOLDER — gerçek: VDevice.create_from_hef(self.hef_path)
            if self.hef_path:
                pass
        except ImportError:
            pass  # sessizce HSV'ye düş

    def classify(self, samples_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._available:
            return self._fallback.classify(samples_bgr)
        raise NotImplementedError("Hailo inference H5'te eklenecek")


def sample_active_row(
    frame_bgr: np.ndarray,
    H_cam_to_chart: np.ndarray,
    rows: int,
    cols: int,
    active_row: int,
    cell_px: int = CELL_PX,
) -> np.ndarray:
    """Verilen sıranın her hücresinin merkez rengini örnekle. (cols, 3) BGR döndür."""
    if not (0 <= active_row < rows):
        return np.empty((0, 3), dtype=np.uint8)

    # Kamerayı chart-piksele warp et (progress.warp ile aynı; bağımsızlık için tekrar).
    scale = np.array(
        [[cell_px, 0, 0], [0, cell_px, 0], [0, 0, 1]], dtype=np.float64,
    )
    M = scale @ np.asarray(H_cam_to_chart, dtype=np.float64)
    out_size = (cols * cell_px, rows * cell_px)
    warped = cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)

    band = warped[active_row * cell_px:(active_row + 1) * cell_px]

    # Hücrenin orta %50'sini örnekle: kenarlarda ızgara çizgisi ve komşu
    # hücre taşması var, orta kısım daha temiz.
    pad = cell_px // 4

    samples = np.empty((cols, 3), dtype=np.uint8)
    for c in range(cols):
        block = band[pad:cell_px - pad, c * cell_px + pad:(c + 1) * cell_px - pad]
        # Bloğun ortalama rengi = hücrenin "ana rengi".
        samples[c] = block.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return samples


def last_completed_row(active_row: int, rows: int, direction: str) -> int | None:
    """Aktif sıranın bir önceki (yön bağımlı) tamamlanmış sırasını döndür.

    Aktif sıra dokumanın en başındaysa böyle bir sıra yoktur → None.
    """
    # bottom_up'ta yapılmış sıra altta (+1), top_down'da üstte (-1).
    r = active_row + 1 if direction == "bottom_up" else active_row - 1
    return r if 0 <= r < rows else None


def check_active_row(
    frame_bgr: np.ndarray,
    H_cam_to_chart: np.ndarray,
    chart,
    active_row: int,
    backend: ColorBackend,
) -> list[tuple[int, int, int, float]]:
    """Verilen sıranın renklerini palette ile karşılaştır.

    Döndürür: uyuşmayan hücreler [(col, expected_idx, observed_idx, lab_distance), ...].
    """
    if not (0 <= active_row < chart.rows):
        return []

    samples = sample_active_row(
        frame_bgr, H_cam_to_chart, chart.rows, chart.cols, active_row,
    )
    if len(samples) == 0:
        return []

    observed_idx, observed_dist = backend.classify(samples)
    expected_idx = chart.grid[active_row]

    mismatches = []
    for c in range(chart.cols):
        if observed_idx[c] != expected_idx[c]:
            # int/float cast: numpy scalar'ları JSON/print uyumlu Python tipine çevir.
            mismatches.append(
                (c, int(expected_idx[c]), int(observed_idx[c]), float(observed_dist[c]))
            )
    return mismatches
