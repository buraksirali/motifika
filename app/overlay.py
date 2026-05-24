"""AR overlay: chart şablonunu canlı kamera görüntüsüne yansıtır.

Kameradaki kilim görüntüsünün üstüne yarı saydam chart çizer. Üç bölge:
  - Tamamlanmış sıralar: düşük opaklık (dokumayı kapatmasın)
  - Aktif sıra: sarı vurgu çerçevesi
  - Yapılacak sıralar: yüksek opaklık (motif net görünür)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import cv2
import numpy as np


# Bir hücrenin chart-piksel boyutu. Diğer modüllerdeki CELL_PX'lerle senkron olmalı.
CELL_PX = 16

# Saydamlık (transparency): kullanıcı 0.10–1.00 arası ayarlar; YÜKSEK = daha şeffaf.
# Temel opaklık = 1 - transparency. Üç bölge bu temelin oranlarıyla çizilir, böylece
# kullanıcı tek değerle tüm motifin görünürlüğünü ayarlar (yapılmış/aktif farkı korunur).
DEFAULT_TRANSPARENCY = 0.60   # %60 → temel opaklık 0.40
DONE_RATIO = 0.36             # yapılmış sıralar — temelin ~%36'sı (soluk)
ACTIVE_RATIO = 1.18           # aktif sıra — temelin ~%118'i (en vurgulu, 1.0'da sınırlı)


@dataclass
class Chart:
    """Palette + grid'i bir arada tutan veri sınıfı."""
    rows: int
    cols: int
    palette_rgb: np.ndarray   # (k, 3) uint8 — RGB renk paleti
    grid: np.ndarray          # (rows, cols) int — palette indeksi
    source: str = ""

    @property
    def palette_bgr(self) -> np.ndarray:
        """RGB paleti BGR'a çevir (cv2 ile çizmeden önce gerekli)."""
        return self.palette_rgb[:, ::-1]

    @classmethod
    def load(cls, path: Path) -> "Chart":
        """chart.json dosyasından Chart üret."""
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
    """AR yansıtmayı yapan sınıf. Chart bitmap'i bir kez kurulur, her karede warp edilir."""
    chart: Chart
    cell_px: int = CELL_PX
    direction: str = "bottom_up"
    transparency: float = DEFAULT_TRANSPARENCY   # 0.10–1.00; yüksek = daha şeffaf
    _chart_layer: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Chart bitmap'i ağır hesap — bir kez hazırla.
        self._chart_layer = self._build_chart_layer()

    def _build_chart_layer(self) -> np.ndarray:
        """Chart pikseli boyutunda, ızgaralı, BGR şablon."""
        cell = self.cell_px
        # Fancy indexing → her hücreye rengini koy, sonra cell x cell bloğa şişir.
        layer = self.chart.palette_bgr[self.chart.grid]
        layer = np.repeat(np.repeat(layer, cell, axis=0), cell, axis=1)
        # np.repeat ardışık olmayan bellek dönebilir; cv2.line ardışık bellek ister.
        # Bu satır olmazsa cv2 bazen "Layout incompatible" hatası verir.
        layer = np.ascontiguousarray(layer)

        h, w = layer.shape[:2]
        # Izgara çizgileri (koyu gri).
        for r in range(self.chart.rows + 1):
            cv2.line(layer, (0, r * cell), (w, r * cell), (40, 40, 40), 1)
        for c in range(self.chart.cols + 1):
            cv2.line(layer, (c * cell, 0), (c * cell, h), (40, 40, 40), 1)
        return layer

    def _alpha_mask(self, active_row: int) -> np.ndarray:
        """Her chart pikseline alpha (0..1) ata. Tek kanal float32 harita.

        self.transparency'den türetilir: temel opaklık = 1 - transparency.
        yapılmış sıralar soluk (DONE_RATIO), aktif sıra vurgulu (ACTIVE_RATIO).
        """
        cell = self.cell_px
        h, w = self.chart.rows * cell, self.chart.cols * cell
        alpha = np.zeros((h, w), dtype=np.float32)

        base = max(0.0, 1.0 - self.transparency)   # todo/temel opaklık
        done_a = base * DONE_RATIO
        active_a = min(base * ACTIVE_RATIO, 1.0)
        todo_a = base

        for r in range(self.chart.rows):
            if self.direction == "bottom_up":
                # bottom_up: aktif sıranın altı = dokunmuş.
                done = r > active_row
            else:
                # top_down: aktif sıranın üstü = dokunmuş.
                done = r < active_row
            if r == active_row:
                a = active_a
            elif done:
                a = done_a
            else:
                a = todo_a
            alpha[r * cell:(r + 1) * cell] = a
        return alpha

    def render(
        self,
        frame_bgr: np.ndarray,
        H_chart_to_cam: np.ndarray,
        active_row: int,
    ) -> np.ndarray:
        """Kameranın üstüne chart'ı yansıtıp aktif sıra çerçevesini çizer."""
        cell = self.cell_px

        # chart_layer chart-PİKSEL boyutunda; H ise chart-BİRİM bekliyor.
        # scale_inv (piksel→birim) ile H'yi birleştirip tek warp yapıyoruz.
        scale_inv = np.array(
            [[1 / cell, 0, 0], [0, 1 / cell, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        # @ matris çarpımı (* element-wise olur, karıştırma).
        M = np.asarray(H_chart_to_cam, dtype=np.float64) @ scale_inv

        # cv2 boyut sırası (W, H).
        out_size = (frame_bgr.shape[1], frame_bgr.shape[0])

        warped_layer = cv2.warpPerspective(self._chart_layer, M, out_size, flags=cv2.INTER_LINEAR)
        # Alpha haritasını da AYNI dönüşümle warp et, yoksa renk/alpha kayar.
        alpha = self._alpha_mask(active_row)
        warped_alpha = cv2.warpPerspective(alpha, M, out_size, flags=cv2.INTER_LINEAR)
        warped_alpha = warped_alpha[..., None]  # (H,W) → (H,W,1), broadcast için

        # Per-pixel alpha blend.
        blended = warped_layer.astype(np.float32) * warped_alpha + \
                  frame_bgr.astype(np.float32) * (1.0 - warped_alpha)
        out = np.clip(blended, 0, 255).astype(np.uint8)

        self._draw_active_row_border(out, H_chart_to_cam, active_row)
        return out

    def _draw_active_row_border(
        self, frame_bgr: np.ndarray, H_chart_to_cam: np.ndarray, active_row: int,
    ) -> None:
        """Aktif sıranın sarı çerçevesini her şeyin üstüne çiz."""
        if not (0 <= active_row < self.chart.rows):
            return

        r = active_row
        # Aktif sıranın 4 köşesi (chart-birim): sol üst, sağ üst, sağ alt, sol alt.
        pts_chart = np.array(
            [[0, r], [self.chart.cols, r], [self.chart.cols, r + 1], [0, r + 1]],
            dtype=np.float32,
        )
        # Homojen koordinat: (x,y) → (x,y,1), H ile çarp, w'ye böl.
        pts_h = np.hstack([pts_chart, np.ones((4, 1), dtype=np.float32)])
        pts_cam = (H_chart_to_cam @ pts_h.T).T
        pts_cam = pts_cam[:, :2] / pts_cam[:, 2:3]

        cv2.polylines(
            frame_bgr,
            [pts_cam.astype(np.int32)],
            isClosed=True,
            color=(0, 255, 255),  # sarı (BGR)
            thickness=3,
        )


def _cli_smoke():
    """Smoke test: chart + kalibrasyon + sahte kamera karesi → overlay PNG."""
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
        frame = np.full((h, w, 3), 200, dtype=np.uint8)  # sahte açık gri kare

    renderer = OverlayRenderer(chart, direction=args.direction)
    out = renderer.render(frame, H_chart_to_cam, active_row=args.active_row)
    cv2.imwrite(str(args.out), out)
    print(f"yazıldı: {args.out}")


if __name__ == "__main__":
    _cli_smoke()
