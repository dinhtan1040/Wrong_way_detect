"""Chia train/val và tạo data.yaml.
Mặc định chia THEO CLIP (khuyến nghị, cần >= 2 clip):
    python split_by_clip.py dataset_all dataset --target runs/detect/train-7/weights/best.pt --val-clips rain_03 --test-clips Wrong_way_2
Chỉ có 1 clip? Chia THEO THỜI GIAN trong clip (đoạn cuối làm val, chừa vùng đệm ở ranh giới):
    python split_by_clip.py dataset_all dataset --target runs/detect/train-7/weights/best.pt --time-split
Ảnh không có file .txt => tạo nhãn rỗng (ảnh nền, dùng để dạy model bỏ qua cột/biển báo).
"""
import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--target", required=True, help="weights model gốc (lấy danh sách lớp)")
ap.add_argument("--val-clips", nargs="*", help="tên clip làm tập val; bỏ trống = chọn ngẫu nhiên")
ap.add_argument("--val-ratio", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--test-clips", nargs="*", default=[],
                help="clip để RIÊNG làm tập test (không dùng khi train/val), đánh giá bằng yolo val split=test")
ap.add_argument("--time-split", action="store_true",
                help="chia theo thời gian TRONG từng clip (dùng khi chỉ có 1 clip)")
ap.add_argument("--gap", type=int, default=3,
                help="--time-split: số ảnh bỏ ở ranh giới train/val để tránh frame kề nhau giống hệt")
a = ap.parse_args()

src, dst = Path(a.src), Path(a.dst)
imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not imgs:
    raise SystemExit("Không có ảnh trong " + str(src))
clip_of = lambda p: p.stem.rsplit("__", 1)[0]


def frame_idx(p):
    try:
        return int(p.stem.rsplit("__", 1)[1])
    except (IndexError, ValueError):
        return 0


clips = sorted({clip_of(p) for p in imgs})
split_of = {}  # ảnh -> "train" | "val" | None (bỏ)

if a.time_split:
    if a.test_clips:
        raise SystemExit("--test-clips chỉ dùng khi chia theo clip (bỏ --time-split).")
    for c in clips:
        ps = sorted((p for p in imgs if clip_of(p) == c), key=lambda p: (frame_idx(p), p.name))
        k = int(len(ps) * (1 - a.val_ratio))
        for i, p in enumerate(ps):
            split_of[p] = "train" if i < k - a.gap else ("val" if i >= k else None)
    mode = "theo THỜI GIAN trong clip"
else:
    test = set(a.test_clips)
    bad = test - set(clips)
    if bad:
        raise SystemExit(f"Không có clip test: {bad}. Clip hiện có: {clips}")
    pool = [c for c in clips if c not in test]
    if len(pool) < 2:
        raise SystemExit(f"Sau khi để riêng clip test chỉ còn {len(pool)} clip cho train/val (cần >= 2). Clip hiện có: {clips}\n"
                         "Hãy cắt thêm frame từ clip khác, hoặc thêm --time-split để chia theo thời gian.")
    if a.val_clips:
        val = set(a.val_clips)
        bad = val - set(pool)
        if bad:
            raise SystemExit(f"Clip val không hợp lệ (không tồn tại hoặc trùng clip test): {bad}. Clip dùng được: {pool}")
    else:
        sh = pool[:]
        random.Random(a.seed).shuffle(sh)
        val = set(sh[:max(1, round(len(sh) * a.val_ratio))])
    if val == set(pool):
        raise SystemExit("Tập val không được chiếm hết clip train/val.")
    for p in imgs:
        c = clip_of(p)
        split_of[p] = "test" if c in test else ("val" if c in val else "train")
    mode = f"theo CLIP (val: {sorted(val)}, test: {sorted(test)})"

if not any(v == "train" for v in split_of.values()) or not any(v == "val" for v in split_of.values()):
    raise SystemExit("Một trong hai tập train/val đang rỗng: cần thêm ảnh hoặc chỉnh --val-ratio/--gap.")

if dst.exists():
    shutil.rmtree(dst)
cnt = {"train": 0, "val": 0, "test": 0}
empty = dropped = 0
for p in imgs:
    sp = split_of[p]
    if sp is None:
        dropped += 1
        continue
    (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst / "images" / sp / p.name)
    lab = src / f"{p.stem}.txt"
    out = dst / "labels" / sp / f"{p.stem}.txt"
    if lab.exists():
        shutil.copy2(lab, out)
        if lab.stat().st_size == 0:
            empty += 1
    else:
        out.write_text("")
        empty += 1
    cnt[sp] += 1

names = YOLO(a.target).names
yaml = [f"path: {dst.resolve().as_posix()}", "train: images/train", "val: images/val"]
if cnt["test"]:
    yaml.append("test: images/test")
yaml.append("names:")
yaml += [f"  {k}: {v}" for k, v in sorted(names.items())]
(dst / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
print(f"Chia {mode}: train {cnt['train']} ảnh | val {cnt['val']} ảnh | test {cnt['test']} ảnh | ảnh nền/rỗng {empty}"
      + (f" | bỏ vùng đệm {dropped}" if dropped else ""))
print(f"Đã ghi {dst / 'data.yaml'}")