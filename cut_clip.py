"""Cắt một đoạn ngắn từ video dài để thử/chẩn đoán nhanh.

Chạy:
    python cut_clip.py --video data/videos/Wrong_way_1.mp4 --start 410 --end 460

Kết quả mặc định: data/clips/<cùng tên video>.mp4 (cùng TÊN FILE để dùng lại configs/<tên>.json).
Lưu ý: thời gian trong clip tính lại từ 0, ví dụ giây 420 của video gốc = giây 10 của clip (nếu --start 410).
Nên chừa dư vài giây trước và sau sự kiện để tracker kịp "làm nóng".
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True, help="giây bắt đầu")
    ap.add_argument("--end", type=float, required=True, help="giây kết thúc")
    ap.add_argument("--out", help="file kết quả (mặc định data/clips/<tên video>.mp4)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0, f1 = int(a.start * fps), int(a.end * fps)
    if n > 0 and f0 >= n:
        raise SystemExit(f"--start vượt quá độ dài video ({n / fps:.1f}s)")
    out = Path(a.out) if a.out else Path("data/clips") / (Path(a.video).stem + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() == Path(a.video).resolve():
        raise SystemExit("File kết quả trùng file gốc, hãy chọn --out khác.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cnt = 0
    for _ in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(fr)
        cnt += 1
    cap.release()
    wr.release()
    print(f"Đã cắt {cnt} khung ({cnt / fps:.1f}s) từ {a.start}s đến {a.start + cnt / fps:.1f}s -> {out}")
    print(f"Giây X của video gốc = giây (X - {a.start:g}) của clip.")


if __name__ == "__main__":
    main()
