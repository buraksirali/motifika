"""Atkı cephesi tespiti — kullanıcının dokuduğu son sıra otomatik bulunur.

Yaklaşım:
  1. Kamera karesini homography ile chart koordinatlarına warp et.
     Çıktı boyutu: (cols * CELL, rows * CELL).
  2. Her sıra için 'dokunmuşluk skoru' hesapla:
     - Dokunmuş satırda renkli iplikler var → yüksek renk varyansı + koyu
     - Dokunmamış satırda yalnızca açık çözgü ipliği görünür → düşük varyans
  3. Skoru tepeden tarayıp dokunmuş→dokunmamış geçiş satırını bul.
  4. EMA + histerezis ile titremeyi azalt; manuel +/- düzeltmesi serbest.

API:
    tracker = ProgressTracker(rows, cols)
    active_row = tracker.update(frame_bgr, H_cam_to_chart)
    tracker.bump(+1) / tracker.bump(-1) / tracker.set(row)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CELL_PX = 12
EMA_ALPHA = 0.4
THRESH_RATIO = 0.55


@dataclass
class ProgressTracker:
    rows: int
    cols: int
    cell_px: int = CELL_PX
    ema_alpha: float = EMA_ALPHA
    thresh_ratio: float = THRESH_RATIO
    direction: str = "bottom_up"  # 'bottom_up' (kilim) | 'top_down'

    def __post_init__(self):
        self._score_ema: np.ndarray | None = None
        self._active_row: int = 0
        self._manual_delta: int = 0

    def warp(self, frame_bgr: np.ndarray, H_cam_to_chart: np.ndarray) -> np.ndarray:
        """Kamera karesini chart koordinatlarına yerleştir (cols*CELL × rows*CELL)."""
        scale = np.array(
            [[self.cell_px, 0, 0],
             [0, self.cell_px, 0],
             [0, 0, 1]],
            dtype=np.float64,
        )
        M = scale @ np.asarray(H_cam_to_chart, dtype=np.float64)
        out_size = (self.cols * self.cell_px, self.rows * self.cell_px)
        return cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)

    def row_scores(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Her sıra için dokunmuşluk skoru (0..1, yüksek = dokunmuş)."""
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        h = self.cell_px
        # Per-row blok: yatay yöndeki varyans (renk farklılığı = doku)
        scores = np.empty(self.rows, dtype=np.float32)
        for r in range(self.rows):
            band = gray[r * h:(r + 1) * h]
            # Yatay (sütun yönünde) std + dikey ortalamaya göre koyuluk
            std_h = float(band.std())
            darkness = 1.0 - float(band.mean()) / 255.0
            scores[r] = 0.6 * std_h / 80.0 + 0.4 * darkness
        scores = np.clip(scores, 0.0, 1.0)
        return scores

    def update(self, frame_bgr: np.ndarray, H_cam_to_chart: np.ndarray) -> int:
        warped = self.warp(frame_bgr, H_cam_to_chart)
        scores = self.row_scores(warped)

        if self._score_ema is None:
            self._score_ema = scores.copy()
        else:
            self._score_ema = (
                self.ema_alpha * scores + (1.0 - self.ema_alpha) * self._score_ema
            )

        peak = float(self._score_ema.max())
        if peak < 0.05:
            auto_row = self.rows if self.direction == "bottom_up" else 0
        else:
            threshold = peak * self.thresh_ratio
            woven = self._score_ema >= threshold
            if self.direction == "bottom_up":
                # alttan yukarı: dokunmuş bölge altta; aktif sıra = en üstteki dokunmuşun bir üstü
                idxs = np.where(woven)[0]
                auto_row = int(idxs.min()) - 1 if len(idxs) else self.rows
                auto_row = max(-1, auto_row)
            else:
                # yukarıdan aşağı: dokunmuş bölge üstte; aktif sıra = ilk dokunmamış
                idxs = np.where(~woven)[0]
                auto_row = int(idxs.min()) if len(idxs) else self.rows

        self._active_row = max(0, min(self.rows, auto_row + self._manual_delta))
        return self._active_row

    def bump(self, delta: int) -> int:
        self._manual_delta += delta
        self._active_row = max(0, min(self.rows, self._active_row + delta))
        return self._active_row

    def set(self, row: int) -> int:
        row = max(0, min(self.rows, row))
        self._manual_delta += row - self._active_row
        self._active_row = row
        return self._active_row

    @property
    def active_row(self) -> int:
        return self._active_row

    def reset_manual(self) -> None:
        self._manual_delta = 0


def _cli_smoke():
    """Hızlı smoke test: bir görsel + 4 köşe → score grafiği."""
    import argparse, json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("assets/calibration.json"))
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text())
    H_cam_to_chart = np.array(cal["H_cam_to_chart"], dtype=np.float32)

    frame = cv2.imread(str(args.image))
    tracker = ProgressTracker(rows=cal["rows"], cols=cal["cols"])
    row = tracker.update(frame, H_cam_to_chart)
    print(f"aktif sıra: {row} / {cal['rows']}")
    print(f"skorlar: {np.round(tracker._score_ema, 2).tolist()}")


if __name__ == "__main__":
    _cli_smoke()
