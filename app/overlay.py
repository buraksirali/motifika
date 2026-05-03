"""AR overlay: chart şablonunu canlı kamera görüntüsüne yansıtır.

Üç bölge:
  - Tamamlanmış sıralar: düşük opaklık (zarif rehber, dokumayı kapatmasın)
  - Aktif sıra: sarı vurgu çerçevesi (kullanıcı buradasın)
  - Yapılacak sıralar: yüksek opaklık (motif net görünür)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import cv2
import numpy as np


CELL_PX = 16
DONE_ALPHA = 0.20
TODO_ALPHA = 0.55
ACTIVE_ALPHA = 0.65


@dataclass
class Chart:
    rows: int
    cols: int
    palette_rgb: np.ndarray   # (k, 3) uint8 — RGB
    grid: np.ndarray          # (rows, cols) int — palette indeksi
    source: str = ""

    @property
    def palette_bgr(self) -> np.ndarray:
        return self.palette_rgb[:, ::-1]

    @classmethod
    def load(cls, path: Path) -> "Chart":
        data = json.loads(Path(path).read_text())
        return cls(
            rows=data["rows"],
            cols=data["cols"],
            palette_rgb=np.array(data["palette"], dtype=np.uint8),
            grid=np.array(data["grid"], dtype=int),
            source=data.get("source", ""),
        )


@dataclass
class OverlayRenderer:
    chart: Chart
    cell_px: int = CELL_PX
    direction: str = "bottom_up"
    _chart_layer: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._chart_layer = self._build_chart_layer()

    def _build_chart_layer(self) -> np.ndarray:
        """Chart pikseli boyutunda, ızgaralı, BGR şablon."""
        cell = self.cell_px
        layer = self.chart.palette_bgr[self.chart.grid]
        layer = np.repeat(np.repeat(layer, cell, axis=0), cell, axis=1)
        layer = np.ascontiguousarray(layer)
        h, w = layer.shape[:2]
        for r in range(self.chart.rows + 1):
            cv2.line(layer, (0, r * cell), (w, r * cell), (40, 40, 40), 1)
        for c in range(self.chart.cols + 1):
            cv2.line(layer, (c * cell, 0), (c * cell, h), (40, 40, 40), 1)
        return layer

    def _alpha_mask(self, active_row: int) -> np.ndarray:
        """Her chart pikseline alpha (0..1) ata; aktif sıra çerçevesi dahil."""
        cell = self.cell_px
        h, w = self.chart.rows * cell, self.chart.cols * cell
        alpha = np.zeros((h, w), dtype=np.float32)

        if self.direction == "bottom_up":
            # done = aktif sıranın altı; todo = aktif sıranın üstü; aktif sıra = vurgu
            for r in range(self.chart.rows):
                if r > active_row:
                    a = DONE_ALPHA
                elif r == active_row:
                    a = ACTIVE_ALPHA
                else:
                    a = TODO_ALPHA
                alpha[r * cell:(r + 1) * cell] = a
        else:
            for r in range(self.chart.rows):
                if r < active_row:
                    a = DONE_ALPHA
                elif r == active_row:
                    a = ACTIVE_ALPHA
                else:
                    a = TODO_ALPHA
                alpha[r * cell:(r + 1) * cell] = a

        # Aktif sıra çerçevesi (sarı kenarlık) alpha=1 yapılır; renk render'da basılır
        return alpha

    def render(
        self,
        frame_bgr: np.ndarray,
        H_chart_to_cam: np.ndarray,
        active_row: int,
    ) -> np.ndarray:
        cell = self.cell_px
        scale_inv = np.array(
            [[1 / cell, 0, 0],
             [0, 1 / cell, 0],
             [0, 0, 1]],
            dtype=np.float64,
        )
        # chart_px → kamera: H_chart_to_cam (chart_unit→cam) ∘ scale_inv (chart_px→chart_unit)
        M = np.asarray(H_chart_to_cam, dtype=np.float64) @ scale_inv

        out_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        warped_layer = cv2.warpPerspective(self._chart_layer, M, out_size, flags=cv2.INTER_LINEAR)

        alpha = self._alpha_mask(active_row)
        warped_alpha = cv2.warpPerspective(alpha, M, out_size, flags=cv2.INTER_LINEAR)
        warped_alpha = warped_alpha[..., None]

        blended = warped_layer.astype(np.float32) * warped_alpha + \
                  frame_bgr.astype(np.float32) * (1.0 - warped_alpha)
        out = np.clip(blended, 0, 255).astype(np.uint8)

        # Aktif sıra çerçevesini direkt kameraya çiz (4 köşeli polygon)
        self._draw_active_row_border(out, H_chart_to_cam, active_row)
        return out

    def _draw_active_row_border(
        self, frame_bgr: np.ndarray, H_chart_to_cam: np.ndarray, active_row: int,
    ) -> None:
        if not (0 <= active_row < self.chart.rows):
            return
        # chart-unit koordinatlarda aktif sıranın 4 köşesi
        r = active_row
        pts_chart = np.array(
            [[0, r], [self.chart.cols, r], [self.chart.cols, r + 1], [0, r + 1]],
            dtype=np.float32,
        )
        pts_h = np.hstack([pts_chart, np.ones((4, 1), dtype=np.float32)])
        pts_cam = (H_chart_to_cam @ pts_h.T).T
        pts_cam = pts_cam[:, :2] / pts_cam[:, 2:3]
        cv2.polylines(
            frame_bgr,
            [pts_cam.astype(np.int32)],
            isClosed=True,
            color=(0, 255, 255),
            thickness=3,
        )


def _cli_smoke():
    """Smoke: chart + kalibrasyon + sahte kamera karesi → overlay PNG."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=Path("assets/calibration.json"))
    ap.add_argument("--frame", type=Path, default=None, help="kullanılırsa kamera kare olarak okunur")
    ap.add_argument("--active-row", type=int, default=10)
    ap.add_argument("--direction", default="bottom_up")
    ap.add_argument("--out", type=Path, default=Path("/tmp/overlay_test.png"))
    args = ap.parse_args()

    chart = Chart.load(args.chart)
    cal = json.loads(args.calibration.read_text())
    H_chart_to_cam = np.array(cal["H_chart_to_cam"], dtype=np.float64)

    if args.frame:
        frame = cv2.imread(str(args.frame))
    else:
        w, h = cal["frame_size"]
        frame = np.full((h, w, 3), 200, dtype=np.uint8)

    renderer = OverlayRenderer(chart, direction=args.direction)
    out = renderer.render(frame, H_chart_to_cam, active_row=args.active_row)
    cv2.imwrite(str(args.out), out)
    print(f"yazıldı: {args.out}")


if __name__ == "__main__":
    _cli_smoke()
