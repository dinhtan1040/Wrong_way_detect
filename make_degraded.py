"""Tạo video suy giảm chất lượng để đo ảnh hưởng của nhiễu / mờ / thiếu sáng / độ phân giải / nén.
Kích thước và fps giữ nguyên nên cấu hình làn dùng lại được.

Chạy (ví dụ):
    python make_degraded.py --video-dir data/videos --out data/videos_noise20 --mode noise --level 20
    python make_degraded.py --video-dir data/videos --out data/videos_dark40  --mode dark  --level 40
    python make_degraded.py --video-dir data/videos --out data/videos_lowres50 --mode lowres --level 50
    python make_degraded.py --video-dir data/videos --out data/videos_jpeg15 --mode jpeg --level 15

mode / level:
    noise  : nhiễu Gauss, level = độ lệch chuẩn (0-255)
    blur   : làm mờ, level = kích thước kernel (số lẻ)
    dark   : tối đi, level = % độ sáng còn lại (40 = còn 40%)
    lowres : giảm độ phân giải rồi phóng lại, level = % kích thước (50 = một nửa)
    jpeg   : nén JPEG mạnh, level = chất lượng 1-100 (càng thấp càng xấu)
Sau đó chạy detect_wrong_way.py + evaluate.py trên thư mục mới, dùng cùng gt.csv.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def degrade(frame, mode, level):
    h, w = frame.shape[:2]
    if mode == "noise":
        return np.clip(frame.astype(np.float32) + np.random.normal(0, level, frame.shape), 0, 255).astype(np.uint8)
    if mode == "blur":
        k = int(level) | 1
        return cv2.GaussianBlur(frame, (k, k), 0)
    if mode == "dark":
        return np.clip(frame.astype(np.float32) * level / 100.0, 0, 255).astype(np.uint8)
    if mode == "lowres":
        s = level / 100.0
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == "jpeg":
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["noise", "blur", "dark", "lowres", "jpeg"])
    ap.add_argument("--level", type=float, required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for vp in sorted(Path(a.video_dir).iterdir()):
        if vp.suffix.lower() not in VIDEO_EXT:
            continue
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        wr = cv2.VideoWriter(str(out / (vp.stem + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            wr.write(degrade(fr, a.mode, a.level))
        cap.release()
        wr.release()
        print("ok", vp.name)


if __name__ == "__main__":
    main()
