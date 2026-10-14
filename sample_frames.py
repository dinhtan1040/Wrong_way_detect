"""Cắt frame từ các clip để gán nhãn, có thể chỉ lấy một đoạn thời gian.

Ví dụ:
    # mọi clip: lấy từ giây 30 đến giây 300
    python sample_frames.py data/videos dataset_all --every-s 0.5 --start 30 --end 300
    # thời gian viết kiểu phút:giây cũng được
    python sample_frames.py data/videos dataset_all --start 1:30 --end 7:00
    # mỗi clip một đoạn riêng (ưu tiên hơn --start/--end); có thể lặp nhiều lần
    python sample_frames.py data/videos dataset_all --range Rain_1=0:30-5:00 --range Rain_2=60-240
    # bỏ đoạn có xe ngược chiều, giữ riêng để test sự kiện (không cắt frame ở đó)
    python sample_frames.py data/videos dataset_all --range Wrong_way_1=0-400 --range Wrong_way_1=460-600
    # clip bỏ hẳn:  --skip Wrong_way_2

Tên ảnh: <clip>__<số khung>.jpg  (tiền tố clip dùng để chia train/val THEO CLIP).
Một clip có thể có nhiều đoạn (lặp --range cho cùng tên clip). Không đặt --range/--start/--end thì lấy cả clip.
--skip: clip giữ riêng làm tập TEST cuối cùng, không cắt frame để huấn luyện.
Lưu ý: KHÔNG lấy frame trong đoạn có xe ngược chiều dùng để đánh giá sự kiện (sẽ làm kết quả bị rò rỉ).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2

EXT = {".mp4", ".mov", ".avi", ".mkv"}


def parse_t(s):
    """'90' -> 90.0 ; '1:30' -> 90.0 ; '1:02:03' -> 3723.0"""
    parts = [float(x) for x in str(s).strip().split(":")]
    if len(parts) > 3 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"thời gian không hợp lệ: {s}")
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def parse_range(s):
    """'Rain_1=0:30-5:00' -> ('Rain_1', 30.0, 300.0); thiếu đầu hoặc cuối ('Rain_1=60-') nghĩa là tới hết."""
    try:
        name, rng = s.rsplit("=", 1)
        a, b = rng.split("-", 1)
        start = parse_t(a) if a.strip() else 0.0
        end = parse_t(b) if b.strip() else None
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(f"--range cần dạng TÊN=BẮT_ĐẦU-KẾT_THÚC, nhận được: {s}")
    if end is not None and end <= start:
        raise argparse.ArgumentTypeError(f"đoạn rỗng: {s}")
    return Path(name).stem, start, end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--every-s", type=float, default=0.5, help="lấy 1 frame mỗi N giây")
    ap.add_argument("--max-per-clip", type=int, default=0, help="giới hạn số frame mỗi đoạn (0 = không giới hạn)")
    ap.add_argument("--skip", nargs="*", default=[], help="tên clip (không đuôi) bỏ qua")
    ap.add_argument("--start", type=parse_t, default=0.0, help="giây bắt đầu (hoặc mm:ss), áp dụng cho mọi clip")
    ap.add_argument("--end", type=parse_t, default=None, help="giây kết thúc (hoặc mm:ss), mặc định tới hết clip")
    ap.add_argument("--range", type=parse_range, action="append", default=[], metavar="TÊN=BĐ-KT",
                    help="đoạn riêng cho một clip, vd Rain_1=0:30-5:00; lặp lại để lấy nhiều đoạn")
    a = ap.parse_args()

    per_clip = defaultdict(list)
    for name, s, e in a.range:
        per_clip[name].append((s, e))
    if a.end is not None and a.end <= a.start:
        raise SystemExit("--end phải lớn hơn --start")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [v for v in sorted(Path(a.clips).iterdir()) if v.suffix.lower() in EXT and v.stem not in a.skip]
    unknown = set(per_clip) - {v.stem for v in clips}
    if unknown:
        print(f"[cảnh báo] --range cho clip không tồn tại (hoặc bị --skip): {sorted(unknown)}")
    for v in clips:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps if n > 0 else None
        segs = per_clip.get(v.stem) or [(a.start, a.end)]
        total = 0
        for s, e in segs:
            f0 = int(round(s * fps))
            f1 = n if (e is None and n > 0) else (int(round(e * fps)) if e is not None else 10 ** 12)
            if n > 0:
                f1 = min(f1, n)
            if n > 0 and f0 >= n:
                print(f"{v.name}: đoạn {s:g}s-{e if e is not None else 'hết'} vượt quá độ dài clip ({dur:.1f}s), bỏ qua")
                continue
            step = max(1, round(a.every_s * fps))
            if a.max_per_clip and f1 < 10 ** 12:
                step = max(step, (f1 - f0) // a.max_per_clip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            i, saved = f0, 0
            while i < f1:
                ok, f = cap.read()
                if not ok:
                    break
                if (i - f0) % step == 0:
                    cv2.imwrite(str(out / f"{v.stem}__{i:06d}.jpg"), f)
                    saved += 1
                i += 1
            total += saved
            print(f"{v.name}: {saved} frame, đoạn {f0 / fps:.1f}s-{i / fps:.1f}s (cách nhau {step} khung)")
        cap.release()
        if len(segs) > 1:
            print(f"  -> tổng {total} frame cho {v.name}")


if __name__ == "__main__":
    main()