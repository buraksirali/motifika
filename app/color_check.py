"""Aktif sıradaki hücre renklerini çıkar ve beklenen palette ile karşılaştır.

İki backend:
  - HSVBackend (varsayılan): klasik OpenCV, eğitim gerektirmez. Demo bunu kullanır.
  - HailoBackend (placeholder): Hailo-8L .hef ile inference. H5'te eklenecek.

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
LAB_DISTANCE_THRESHOLD = 25.0


class ColorBackend(Protocol):
    def classify(self, samples_bgr: np.ndarray) -> np.ndarray:
        """Verilen N×3 örneği için palet indeksi (N,) döndür."""
        ...


@dataclass
class HSVBackend:
    """LAB uzayında en yakın palet rengini bul (klasik, hızlı, eğitim gerektirmez)."""
    palette_rgb: np.ndarray  # (k, 3) uint8

    def __post_init__(self):
        # palette_rgb -> palette_lab (1, k, 3) ile cv2 uyumlu
        bgr = self.palette_rgb[:, ::-1].astype(np.uint8)[None, :, :]
        self._palette_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0].astype(np.float32)

    def classify(self, samples_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if samples_bgr.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        lab = cv2.cvtColor(samples_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        lab = lab.reshape(-1, 3).astype(np.float32)
        # (N, k) mesafe matrisi
        diff = lab[:, None, :] - self._palette_lab[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        idx = dist.argmin(axis=1)
        min_dist = dist.min(axis=1)
        return idx, min_dist


@dataclass
class HailoBackend:
    """Hailo-8L .hef wrapper. H5'te implementasyon yapılacak; şimdilik HSV'ye düşer."""
    palette_rgb: np.ndarray
    hef_path: str | None = None

    def __post_init__(self):
        self._fallback = HSVBackend(self.palette_rgb)
        self._available = False
        try:
            # import hailo_platform  # noqa: F401
            # gerçek implementasyon: VDevice.create_from_hef(self.hef_path)
            if self.hef_path:
                pass
        except ImportError:
            pass

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
    """Aktif sıranın N hücresinin merkez renklerini örnekle. (cols, 3) BGR döndür."""
    if not (0 <= active_row < rows):
        return np.empty((0, 3), dtype=np.uint8)

    scale = np.array(
        [[cell_px, 0, 0], [0, cell_px, 0], [0, 0, 1]], dtype=np.float64,
    )
    M = scale @ np.asarray(H_cam_to_chart, dtype=np.float64)
    out_size = (cols * cell_px, rows * cell_px)
    warped = cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)

    band = warped[active_row * cell_px:(active_row + 1) * cell_px]
    # Her sütun bloğunun merkez 50% alanının ortalama rengini al (kenar kontaminasyonunu azalt)
    pad = cell_px // 4
    samples = np.empty((cols, 3), dtype=np.uint8)
    for c in range(cols):
        block = band[pad:cell_px - pad, c * cell_px + pad:(c + 1) * cell_px - pad]
        samples[c] = block.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return samples


def last_completed_row(active_row: int, rows: int, direction: str) -> int | None:
    """Aktif sıranın bir önceki (yön bağımlı) tamamlanmış sırasını döndür."""
    if direction == "bottom_up":
        r = active_row + 1
    else:
        r = active_row - 1
    return r if 0 <= r < rows else None


def check_active_row(
    frame_bgr: np.ndarray,
    H_cam_to_chart: np.ndarray,
    chart,
    active_row: int,
    backend: ColorBackend,
    distance_threshold: float = LAB_DISTANCE_THRESHOLD,
) -> list[tuple[int, int, int, float]]:
    """
    Verilen sıranın renklerini palette ile karşılaştır.
    Döndürür: uyuşmayan hücrelerin listesi (col, expected_idx, observed_idx, lab_distance).
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
            mismatches.append(
                (c, int(expected_idx[c]), int(observed_idx[c]), float(observed_dist[c]))
            )
    return mismatches
