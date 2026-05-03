"""Çalışma alanı (kilim ROI) kalibrasyonu.

Kullanıcı kameradan kilim çalışma alanının 4 köşesine sırayla tıklar
(SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT). chart pikseli ↔ kamera pikseli
arasındaki homography matrisleri assets/calibration.json'a kaydedilir.

Kullanım:
    python -m app.calibration --rows 30 --cols 60 --camera 0
    python -m app.calibration --rows 30 --cols 60 --image test.jpg

Çıktı JSON:
    {
        "rows": 30, "cols": 60,
        "camera_corners": [[x,y], ...],         # tıklanan 4 nokta (kamera piksel)
        "frame_size": [w, h],                    # kamera kare boyutu
        "H_chart_to_cam": [[...]],               # 3x3
        "H_cam_to_chart": [[...]]                # 3x3 (ters)
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CORNER_LABELS = ["SOL UST", "SAG UST", "SAG ALT", "SOL ALT"]
DEFAULT_PATH = Path("assets/calibration.json")


def chart_corners(rows: int, cols: int) -> np.ndarray:
    """Chart koordinatlarında köşe noktaları (sol üst sıfır, sağ alt cols x rows)."""
    return np.array(
        [[0, 0], [cols, 0], [cols, rows], [0, rows]],
        dtype=np.float32,
    )


def compute_homography(camera_corners: np.ndarray, rows: int, cols: int):
    src_chart = chart_corners(rows, cols)
    dst_cam = np.asarray(camera_corners, dtype=np.float32)
    H_chart_to_cam = cv2.getPerspectiveTransform(src_chart, dst_cam)
    H_cam_to_chart = np.linalg.inv(H_chart_to_cam)
    return H_chart_to_cam, H_cam_to_chart


def collect_corners_interactive(frame_provider, rows: int, cols: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pencerede 4 köşeye tıklatır. frame_provider() güncel kareyi döndürür."""
    state = {"points": [], "frozen": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 4:
            state["points"].append((x, y))

    win = "MOTIFIKA - Kalibrasyon"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    frame_size = None
    while True:
        if state["frozen"] is not None:
            frame = state["frozen"].copy()
        else:
            frame = frame_provider()
            if frame is None:
                continue
            frame_size = (frame.shape[1], frame.shape[0])

        for i, p in enumerate(state["points"]):
            cv2.circle(frame, p, 8, (0, 255, 255), -1)
            cv2.putText(
                frame, str(i + 1), (p[0] + 10, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        if len(state["points"]) >= 2:
            cv2.polylines(
                frame,
                [np.array(state["points"], np.int32)],
                isClosed=(len(state["points"]) == 4),
                color=(0, 255, 0), thickness=2,
            )

        idx = len(state["points"])
        msg = (
            f"Tikla: {CORNER_LABELS[idx]} ({idx + 1}/4)"
            if idx < 4
            else "ENTER=kaydet  R=sifirla  ESC=iptal"
        )
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(
            frame, f"Izgara: {rows} sira x {cols} sutun",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("Kalibrasyon iptal edildi")
        if key in (ord("r"), ord("R")):
            state["points"].clear()
            state["frozen"] = None
        if key == ord(" ") and state["frozen"] is None:
            state["frozen"] = frame_provider()
        if key in (10, 13) and len(state["points"]) == 4:
            break

    cv2.destroyWindow(win)
    return np.array(state["points"], dtype=np.float32), frame_size


def save_calibration(
    out_path: Path,
    rows: int,
    cols: int,
    camera_corners: np.ndarray,
    frame_size: tuple[int, int],
) -> dict:
    H_chart_to_cam, H_cam_to_chart = compute_homography(camera_corners, rows, cols)
    data = {
        "rows": rows,
        "cols": cols,
        "camera_corners": camera_corners.tolist(),
        "frame_size": list(frame_size),
        "H_chart_to_cam": H_chart_to_cam.tolist(),
        "H_cam_to_chart": H_cam_to_chart.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def load_calibration(path: Path = DEFAULT_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture indeksi")
    ap.add_argument("--image", type=Path, default=None, help="canli kamera yerine sabit goruntu")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
        provider = lambda: img.copy()  # noqa: E731
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: {args.camera}")
        def provider():
            ok, frame = cap.read()
            return frame if ok else None

    try:
        corners, frame_size = collect_corners_interactive(provider, args.rows, args.cols)
    finally:
        if args.image is None:
            cap.release()

    data = save_calibration(args.out, args.rows, args.cols, corners, frame_size)
    print(f"kalibrasyon kaydedildi: {args.out}")
    print(f"köşeler (kamera px): {data['camera_corners']}")


if __name__ == "__main__":
    main()
