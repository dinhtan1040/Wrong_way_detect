"""Nhãn nháp bằng CHÍNH model VisDrone (train-7), không dùng COCO.
    python prelabel_visdrone.py dataset_all --target runs/detect/train-7/weights/best.pt

Gộp lớp VisDrone về các lớp của bài toán: van->car, tricycle/awning-tricycle->motor.
bus/truck đòi conf cao hơn (--big-conf) để không sinh nhãn cho cột/khối chắn.
Ảnh đã có .txt thì bỏ qua (không ghi đè nhãn đã sửa) trừ khi --overwrite.
Lưu ý: model này hay BỎ SÓT xe máy áo mưa => bạn phải tự VẼ BÙ các khung motor còn thiếu khi sửa nhãn.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--target", required=True)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.15)
ap.add_argument("--big-conf", type=float, default=0.5, help="conf tối thiểu cho bus/truck")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

model = YOLO(a.target)
names = {k: n.lower() for k, n in model.names.items()}
id_of = {n: k for k, n in names.items()}
MAP = {"car": "car", "van": "car", "truck": "truck", "bus": "bus",
       "motor": "motor", "tricycle": "motor", "awning-tricycle": "motor"}
for need in {"car", "truck", "bus", "motor"}:
    if need not in id_of:
        raise SystemExit(f"Model không có lớp '{need}'. Lớp có: {list(id_of)}")

imgs = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
done = 0
for p in imgs:
    txt = p.with_suffix(".txt")
    if txt.exists() and not a.overwrite:
        continue
    r = model.predict(str(p), imgsz=a.imgsz, conf=a.conf, iou=0.5, agnostic_nms=True, verbose=False)[0]
    lines = []
    for (x, y, w, h), k, c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        tgt = MAP.get(names[k])
        if tgt is None:
            continue
        if tgt in ("bus", "truck") and c < a.big_conf:
            continue
        lines.append(f"{id_of[tgt]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    done += 1
print(f"Đã tạo nhãn nháp cho {done} ảnh. Hãy sửa nhãn trước khi train.")