"""Bước 5b: gom ảnh + nhãn đã sửa thành dataset YOLO, CHIA THEO VIDEO (không chia theo khung).

Chạy:
    python 04_make_dataset.py --frames data/frames --out dataset \
        --test-videos rain_03 night_02 --val-videos sun_02

- test-videos: video/buổi quay giữ lại hoàn toàn, chỉ dùng để đánh giá cuối.
- val-videos : dùng để chọn checkpoint khi train. Bỏ trống -> tự lấy mỗi 5 video một video.
Tên video là tên thư mục trong data/frames (không có đuôi .mp4).

NOTE: dùng CÙNG --weights với 03_autolabel.py (ví dụ runs/detect/train-7/weights/best.pt) để tên lớp khớp.
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--weights", default="yolo11m.pt", help="model dùng ở 03_autolabel.py (lấy tên lớp)")
    ap.add_argument("--val-videos", nargs="*", default=[])
    ap.add_argument("--test-videos", nargs="*", default=[])
    a = ap.parse_args()

    root, out = Path(a.frames), Path(a.out)
    stems = sorted(d.name for d in root.iterdir() if d.is_dir())
    test = set(a.test_videos)
    val = set(a.val_videos)
    if not val:
        val = set([s for s in stems if s not in test][::5])

    counts = {"train": 0, "val": 0, "test": 0}
    for stem in stems:
        split = "test" if stem in test else "val" if stem in val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((root / stem).glob("*.jpg")):
            shutil.copy2(img, out / "images" / split / img.name)
            lab = img.with_suffix(".txt")
            if lab.exists():
                shutil.copy2(lab, out / "labels" / split / lab.name)
            counts[split] += 1

    names = YOLO(a.weights).names
    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": dict(names)}
    if counts["test"]:
        data["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(counts, "->", out / "data.yaml")
    print("Train:", [s for s in stems if s not in val and s not in test])
    print("Val  :", sorted(val))
    print("Test :", sorted(test))


if __name__ == "__main__":
    main()
