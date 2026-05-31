"""Atkı cephesi tespiti — kullanıcının dokuduğu son sıra otomatik bulunur.

ATKI = kilim dokumada yatay renk ipliği; her sıra bir atkı.
ATKI CEPHESİ = şu an dokunan yer; yukarı doğru ilerler.

Algoritma:
  1. Kamera karesini homography ile chart koordinatlarına warp et.
  2. Her sıra için 'dokunmuşluk skoru' hesapla (renk varyansı + koyuluk).
  3. Skoru tarayıp dokunmuş→dokunmamış geçiş satırını bul.
  4. EMA + relatif eşik ile titremeyi azalt; manuel +/- düzeltme serbest.

API:
    tracker = ProgressTracker(rows, cols)
    active_row = tracker.update(frame_bgr, H_cam_to_chart)
    tracker.bump(+1) / tracker.bump(-1) / tracker.set(row)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Skor hesabı için yeterli çözünürlük (overlay'de 16). 30x60x12² ≈ 1MB float32.
CELL_PX = 12

# EMA katsayısı: yeni_ema = α*yeni + (1-α)*eski. Yüksek = titrek, düşük = geç tepki.
EMA_ALPHA = 0.4

# En yüksek skorun bu oranı üstü "dokunmuş" sayılır. Relatif eşik ışıktan bağımsız.
THRESH_RATIO = 0.55


@dataclass
class ProgressTracker:
    """Aktif sırayı tahmin eden durum makinesi."""
    rows: int
    cols: int
    cell_px: int = CELL_PX
    ema_alpha: float = EMA_ALPHA
    thresh_ratio: float = THRESH_RATIO
    direction: str = "bottom_up"  # 'bottom_up' (kilim) | 'top_down'

    def __post_init__(self):
        self._score_ema: np.ndarray | None = None  # yumuşatılmış skor; ilk update'te dolar
        self._active_row: int = 0
        self._manual_delta: int = 0  # kullanıcının +/- ile eklediği kalıcı offset

    def warp(self, frame_bgr: np.ndarray, H_cam_to_chart: np.ndarray) -> np.ndarray:
        """Kamera karesini chart-piksel uzayına yerleştir (cols*CELL × rows*CELL)."""
        # scale (birim→piksel) ile H'yi birleştir; tek warp.
        scale = np.array(
            [[self.cell_px, 0, 0], [0, self.cell_px, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        M = scale @ np.asarray(H_cam_to_chart, dtype=np.float64)
        out_size = (self.cols * self.cell_px, self.rows * self.cell_px)  # cv2 (W,H)
        return cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)

    def row_scores(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Her sıra için dokunmuşluk skoru (0..1, yüksek = dokunmuş)."""
        # Gri yeter (varyans + parlaklık); 1 kanal = daha hızlı.
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        h = self.cell_px
        scores = np.empty(self.rows, dtype=np.float32)

        for r in range(self.rows):
            band = gray[r * h:(r + 1) * h]
            # std: boş kilim tek tonda (std≈0), dokunmuş bölge çeşitli (std büyür).
            std_h = float(band.std())
            # darkness: dokunmuş ip koyu → yüksek.
            darkness = 1.0 - float(band.mean()) / 255.0
            # Karma skor: çeşitlilik 0.6 ağırlık, koyuluk 0.4. 80 = tipik std normalizasyonu.
            scores[r] = 0.6 * std_h / 80.0 + 0.4 * darkness

        return np.clip(scores, 0.0, 1.0)

    def update(self, frame_bgr: np.ndarray, H_cam_to_chart: np.ndarray) -> int:
        """Bir kareyi işle, güncel aktif sırayı döndür."""
        warped = self.warp(frame_bgr, H_cam_to_chart)
        scores = self.row_scores(warped)

        # EMA ile yumuşat; ilk karede önceki yok → doğrudan kullan.
        if self._score_ema is None:
            self._score_ema = scores.copy()
        else:
            self._score_ema = (
                self.ema_alpha * scores + (1.0 - self.ema_alpha) * self._score_ema
            )

        peak = float(self._score_ema.max())

        if peak < 0.05:
            # Tüm skorlar çok düşük → boş kilim. bottom_up'ta dokuma en ALT
            # sıradan (rows-1) başlar ve yukarı çıkar; top_down'da en üstten (0).
            auto_row = self.rows - 1 if self.direction == "bottom_up" else 0
        else:
            threshold = peak * self.thresh_ratio
            woven = self._score_ema >= threshold  # boolean dizi: dokunmuş mu

            if self.direction == "bottom_up":
                # Dokunmuş bölge altta; aktif = en üstteki dokunmuşun bir üstü.
                idxs = np.where(woven)[0]
                auto_row = int(idxs.min()) - 1 if len(idxs) else self.rows - 1
                auto_row = max(-1, auto_row)
            else:
                # Dokunmuş bölge üstte; aktif = ilk dokunmamış sıra.
                idxs = np.where(~woven)[0]
                auto_row = int(idxs.min()) if len(idxs) else self.rows

        # Manuel offset'i ekle, [0, rows] aralığına sıkıştır.
        self._active_row = max(0, min(self.rows, auto_row + self._manual_delta))
        return self._active_row

    def bump(self, delta: int) -> int:
        """+/- tuşu: manuel offset'i kalıcı olarak değiştir."""
        self._manual_delta += delta
        self._active_row = max(0, min(self.rows, self._active_row + delta))
        return self._active_row

    def set(self, row: int) -> int:
        """Belirli satıra zıpla (fark manuel offset'e işlenir, sonraki update'te korunur)."""
        row = max(0, min(self.rows, row))
        self._manual_delta += row - self._active_row
        self._active_row = row
        return self._active_row

    @property
    def active_row(self) -> int:
        return self._active_row

    def reset_manual(self) -> None:
        """Manuel offset'i sıfırla (yön değişiminde eski offset geçersiz)."""
        self._manual_delta = 0


def _cli_smoke():
    """Hızlı smoke test: bir görsel + 4 köşe → aktif sıra + skorlar."""
    import argparse, json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("assets/calibration.json"))
    ap.add_argument("--rows", type=int, default=44)
    ap.add_argument("--cols", type=int, default=38)
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text())
    # Izgaradan bağımsız birim-kare kalibrasyonunu rows×cols'a göre normalize et.
    H_unit = np.array(cal["H_unit_to_cam"], dtype=np.float64)
    norm = np.array([[1 / args.cols, 0, 0], [0, 1 / args.rows, 0], [0, 0, 1]], dtype=np.float64)
    H_cam_to_chart = np.linalg.inv(H_unit @ norm).astype(np.float32)

    frame = cv2.imread(str(args.image))
    tracker = ProgressTracker(rows=args.rows, cols=args.cols)
    row = tracker.update(frame, H_cam_to_chart)
    print(f"aktif sıra: {row} / {args.rows}")
    print(f"skorlar: {np.round(tracker._score_ema, 2).tolist()}")


if __name__ == "__main__":
    _cli_smoke()
