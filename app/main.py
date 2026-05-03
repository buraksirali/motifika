"""MOTİFİKA ana döngü.

Kullanım:
    python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0
    python -m app.main --motif hayat_agaci --rows 50 --cols 30 --image kilim.jpg
    python -m app.main --recalibrate ...

Klavye:
    [+]/[=]   aktif sırayı +1
    [-]       aktif sırayı -1
    [r]       kalibrasyonu sıfırla (yeni 4-köşe)
    [d]       yön değiştir (bottom_up <-> top_down)
    [c]       renk kontrolünü aç/kapat
    [q]/ESC   çıkış
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.calibration import (
    DEFAULT_PATH as CAL_PATH,
    collect_corners_interactive,
    load_calibration,
    save_calibration,
)
from app.color_check import HSVBackend, check_active_row, last_completed_row
from app.overlay import Chart, OverlayRenderer
from app.pattern import build_chart, save_chart
from app.progress import ProgressTracker
from app.ui import UIRenderer


MOTIFS = {
    "eli_belinde": Path("eli_belinde.jpg"),
    "hayat_agaci": Path("hayat_agaci_ornek.png"),
}
COLOR_CHECK_EVERY_N = 5


def ensure_chart(motif: str, rows: int, cols: int, palette: int) -> Path:
    """assets/<motif>_chart.json yoksa motif kaynağından üret."""
    assets = Path("assets")
    chart_path = assets / f"{motif}_chart.json"
    if not chart_path.exists():
        src = MOTIFS.get(motif)
        if src is None or not src.exists():
            raise FileNotFoundError(f"motif kaynağı yok: {motif}")
        chart = build_chart(src, rows, cols, palette)
        save_chart(chart, chart_path)
    return chart_path


def open_camera_or_image(args, frame_size_hint=(1280, 720)):
    """Kamera veya sabit görüntü için frame_provider + cap döndür."""
    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(args.image)
        return (lambda: img.copy()), None, (img.shape[1], img.shape[0])
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"kamera açılamadı: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size_hint[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size_hint[1])

    def provider():
        ok, frame = cap.read()
        return frame if ok else None
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return provider, cap, (real_w, real_h)


def run_calibration_flow(provider, rows, cols, frame_size, out_path: Path) -> dict:
    corners, fs = collect_corners_interactive(provider, rows, cols)
    if fs is None:
        fs = frame_size
    return save_calibration(out_path, rows, cols, corners, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", choices=list(MOTIFS.keys()), default="eli_belinde")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--palette", type=int, default=4)
    ap.add_argument("--direction", choices=["bottom_up", "top_down"], default="bottom_up")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", type=Path, default=None,
                    help="kamera yerine sabit görüntü ile test")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--no-color-check", action="store_true")
    args = ap.parse_args()

    chart_path = ensure_chart(args.motif, args.rows, args.cols, args.palette)
    chart = Chart.load(chart_path)
    print(f"chart yüklendi: {chart_path} ({chart.rows}×{chart.cols}, {len(chart.palette_rgb)} renk)")

    provider, cap, frame_size = open_camera_or_image(args)

    cal_data = load_calibration(CAL_PATH)
    cal_match = (
        cal_data is not None
        and cal_data.get("rows") == args.rows
        and cal_data.get("cols") == args.cols
        and tuple(cal_data.get("frame_size", [])) == frame_size
    )

    try:
        if args.recalibrate or not cal_match:
            print("kalibrasyon başlatılıyor: 4 köşeye SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT sırasıyla tıkla")
            cal_data = run_calibration_flow(provider, args.rows, args.cols, frame_size, CAL_PATH)
            print(f"kalibrasyon kaydedildi: {CAL_PATH}")

        H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
        H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)

        direction = args.direction
        tracker = ProgressTracker(rows=chart.rows, cols=chart.cols, direction=direction)
        renderer = OverlayRenderer(chart, direction=direction)
        ui = UIRenderer()
        backend = HSVBackend(chart.palette_rgb)

        do_color_check = not args.no_color_check
        win = "MOTIFIKA"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        frame_idx = 0
        last_mismatches: list = []
        t_prev = time.time()
        fps_ema = 0.0

        while True:
            frame = provider()
            if frame is None:
                break
            frame_idx += 1

            active_row = tracker.update(frame, H_cam_to_chart)

            if do_color_check and frame_idx % COLOR_CHECK_EVERY_N == 0:
                check_row = last_completed_row(active_row, chart.rows, direction)
                if check_row is not None:
                    last_mismatches = check_active_row(
                        frame, H_cam_to_chart, chart, check_row, backend,
                    )
                else:
                    last_mismatches = []

            ar_view = renderer.render(frame, H_chart_to_cam, active_row)
            ar_view = _resize_to_height(ar_view, target_h=720)

            check_row = last_completed_row(active_row, chart.rows, direction)
            panel = ui.render_panel(
                chart, active_row, direction, last_mismatches,
                height=ar_view.shape[0], check_row=check_row,
            )
            composed = ui.compose(ar_view, panel)

            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_ema = 0.85 * fps_ema + 0.15 * inst_fps if fps_ema else inst_fps
            t_prev = t_now
            cv2.putText(composed, f"{fps_ema:.1f} FPS",
                        (composed.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(win, composed)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("+"), ord("=")):
                tracker.bump(+1)
            elif key in (ord("-"), ord("_")):
                tracker.bump(-1)
            elif key in (ord("r"), ord("R")):
                cv2.destroyWindow(win)
                cal_data = run_calibration_flow(
                    provider, args.rows, args.cols, frame_size, CAL_PATH,
                )
                H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], dtype=np.float64)
                H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], dtype=np.float64)
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            elif key in (ord("d"), ord("D")):
                direction = "top_down" if direction == "bottom_up" else "bottom_up"
                tracker.direction = direction
                renderer.direction = direction
                tracker.reset_manual()
            elif key in (ord("c"), ord("C")):
                do_color_check = not do_color_check
                if not do_color_check:
                    last_mismatches = []
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


if __name__ == "__main__":
    main()
