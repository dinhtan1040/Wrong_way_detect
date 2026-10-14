"""Vẽ cấu hình làn lên khung hình THẬT của video để kiểm tra bằng mắt:
polygon từng làn, tên làn và mũi tên CHIỀU ĐÚNG (chiều xe được phép đi).

Chạy:
    python preview_config.py --video data/videos/Wrong_way_1.mp4 --config configs
    python preview_config.py --video data/videos/Rain_2.mp4 --config configs/Rain_2.json --out prev.png

Kết quả: ảnh PNG gồm 3 khung (đầu / giữa / cuối video). Kiểm tra:
  1. Polygon có phủ đúng vùng xe chạy, đúng làn không (nếu lệch: cấu hình vẽ cho video khác hoặc video bị
     crop/đổi kích thước -> vẽ lại bằng 02_draw_roi.py).
  2. Mũi tên có chỉ đúng chiều xe ĐƯỢC PHÉP đi trong làn đó không. Xe đi ngược mũi tên sẽ bị báo.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from detect_wrong_way import build_lanes, pick_cfg

COLORS = [(0, 200, 0), (0, 140, 255), (255, 120, 0), (200, 0, 200)]


def describe(d):
    """Mô tả chiều (dx, dy) bằng chữ ASCII (cv2.putText không vẽ được tiếng Việt có dấu)."""
    dx, dy = d
    if abs(dy) >= abs(dx):
        return "DOWN (to cam)" if dy > 0 else "UP (from cam)"
    return "RIGHT" if dx > 0 else "LEFT"


def describe_vi(d):
    dx, dy = d
    if abs(dy) >= abs(dx):
        return "đi XUỐNG (về phía camera)" if dy > 0 else "đi LÊN (ra xa camera)"
    return "sang PHẢI" if dx > 0 else "sang TRÁI"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True, help="file JSON hoặc thư mục configs")
    ap.add_argument("--out", help="ảnh kết quả (mặc định: preview_<tên_video>.png)")
    ap.add_argument("--at", type=float, nargs="*", help="xem tại các giây này thay cho đầu/giữa/cuối (ví dụ --at 83.5 141)")
    ap.add_argument("--tile-h", type=int, default=900, help="chiều cao mỗi khung trong ảnh kết quả")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video {a.video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cfg = pick_cfg(a.config, a.video)
    cw, ch = cfg["frame_size"]
    lanes = build_lanes(cfg, W, H)

    print(f"Video: {W}x{H}   |   cấu hình vẽ trên khung: {cw}x{ch}")
    if abs(W / H - cw / ch) > 0.01:
        print("CẢNH BÁO: tỉ lệ khung hình của video KHÁC tỉ lệ lúc vẽ cấu hình -> polygon bị co giãn lệch. "
              "Vẽ lại cấu hình bằng chính video này.")
    elif (W, H) != (cw, ch):
        print(f"Lưu ý: kích thước khác nhưng cùng tỉ lệ, polygon được co giãn đều (x{W / cw:.2f}).")
    for l in lanes:
        print(f"  {l['name']}: chiều đúng = {describe_vi(l['dir'])}  (vector chuẩn hoá {l['dir'].round(2).tolist()})")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    positions = [int(t * fps) for t in a.at] if a.at else [int(max(0, n - 1) * f) for f in (0.05, 0.5, 0.95)]
    tiles = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos))
        ok, fr = cap.read()
        if not ok:
            continue
        over = fr.copy()
        for i, l in enumerate(lanes):
            col = COLORS[i % len(COLORS)]
            pts = l["poly"].astype(np.int32)
            cv2.fillPoly(over, [pts], col)
        fr = cv2.addWeighted(over, 0.25, fr, 0.75, 0)
        thick = max(2, W // 300)
        for i, l in enumerate(lanes):
            col = COLORS[i % len(COLORS)]
            pts = l["poly"].astype(np.int32)
            cv2.polylines(fr, [pts], True, col, thick)
            c = pts.mean(0)
            d = l["dir"] * (H * 0.12)
            cv2.arrowedLine(fr, (int(c[0] - d[0] / 2), int(c[1] - d[1] / 2)),
                            (int(c[0] + d[0] / 2), int(c[1] + d[1] / 2)), (0, 0, 255), thick * 2, tipLength=0.3)
            fs = H / 1400.0
            for k, txt in enumerate((f"{l['name']}", f"dung: {describe(l['dir'])}")):
                p = (int(max(8, min(c[0] - W * 0.22, W - W * 0.5))), int(c[1] + H * (0.1 + 0.04 * k)))
                cv2.putText(fr, txt, p, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(4, int(fs * 6)))
                cv2.putText(fr, txt, p, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(2, int(fs * 2)))
        s = a.tile_h / H
        tiles.append(cv2.resize(fr, (int(W * s), a.tile_h)))
    cap.release()
    if not tiles:
        raise SystemExit("Không đọc được khung hình nào")

    out = a.out or f"preview_{Path(a.video).stem}.png"
    cv2.imwrite(out, np.hstack(tiles))
    print(f"Đã lưu {out}. Mũi tên đỏ = chiều xe được phép đi; xe đi ngược mũi tên sẽ bị báo.")


if __name__ == "__main__":
    main()
