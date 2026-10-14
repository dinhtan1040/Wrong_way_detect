"""Bước 1: trích khung hình từ video tự quay để dán nhãn.

Chạy:
    python 01_extract_frames.py --video-dir data/videos --out data/frames --every-sec 1.5

Mỗi video -> thư mục data/frames/<ten_video>/<ten_video>_<so_khung>.jpg
(lấy mẫu thưa vì các khung kề nhau gần như giống hệt).
"""
import argparse
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--every-sec", type=float, default=1.5, help="lấy 1 khung mỗi N giây")
    ap.add_argument("--max-per-video", type=int, default=400)
    a = ap.parse_args()

    videos = [p for p in sorted(Path(a.video_dir).iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not videos:
        raise SystemExit(f"Không thấy video trong {a.video_dir}")

    total = 0
    for vp in videos:
        out_dir = Path(a.out) / vp.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps * a.every_sec))
        idx = saved = 0
        while saved < a.max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                cv2.imwrite(str(out_dir / f"{vp.stem}_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            idx += 1
        cap.release()
        total += saved
        print(f"{vp.name}: {saved} khung (fps={fps:.1f}, bước={step})")
    print(f"Tổng: {total} khung -> {a.out}")


if __name__ == "__main__":
    main()
