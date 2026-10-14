"""Tạo NHÃN NHÁP (YOLO .txt) bằng model COCO để bạn chỉ phải SỬA thay vì vẽ từ đầu.
    python prelabel.py dataset_all --coco yolo11m.pt --target runs/detect/train-7/weights/best.pt

Quy tắc: car/truck/bus lấy thẳng; motorcycle + người lái (person) gộp thành MỘT box lớp `motor`;
person đứng riêng (conf >= --person-conf) coi là người lái xe máy (tắt bằng --no-person-as-motor).
Ảnh đã có .txt thì bỏ qua (không ghi đè nhãn bạn đã sửa) trừ khi --overwrite.
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--coco", default="yolo11m.pt")
ap.add_argument("--target", required=True, help="weights của model đích (lấy tên lớp/thứ tự id)")
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--veh-conf", type=float, default=0.3)
ap.add_argument("--moto-conf", type=float, default=0.25)
ap.add_argument("--person-conf", type=float, default=0.4)
ap.add_argument("--no-person-as-motor", action="store_true")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

tnames = {n.lower(): k for k, n in YOLO(a.target).names.items()}
for need in ("car", "truck", "bus", "motor"):
    if need not in tnames:
        raise SystemExit(f"Model đích không có lớp '{need}'. Lớp có: {list(tnames)}")
coco = YOLO(a.coco)
PERSON, CAR, MOTO, BUS, TRUCK = 0, 2, 3, 5, 7  # id lớp COCO
VEH = {CAR: "car", BUS: "bus", TRUCK: "truck"}


def center_in(box, outer, pad=0.5):
    x1, y1, x2, y2 = outer
    w, h = x2 - x1, y2 - y1
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return x1 - pad * w <= cx <= x2 + pad * w and y1 - pad * h <= cy <= y2 + pad * h


def union(b1, b2):
    return np.array([min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])])


imgs = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
done = 0
for p in imgs:
    txt = p.with_suffix(".txt")
    if txt.exists() and not a.overwrite:
        continue
    r = coco.predict(str(p), imgsz=a.imgsz, conf=0.1, classes=[PERSON, CAR, MOTO, BUS, TRUCK], verbose=False)[0]
    H, W = r.orig_shape
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    cf = r.boxes.conf.cpu().numpy()
    persons = [(b, c) for b, k, c in zip(xyxy, cls, cf) if k == PERSON]
    used, out = set(), []
    for b, k, c in zip(xyxy, cls, cf):
        if k in VEH and c >= a.veh_conf:
            out.append((tnames[VEH[k]], b))
        elif k == MOTO:
            matched = [i for i, (pb, _) in enumerate(persons) if center_in(pb, b)]
            if c >= a.moto_conf or matched:
                box = b.copy()
                for i in matched:
                    box = union(box, persons[i][0])
                    used.add(i)
                out.append((tnames["motor"], box))
    if not a.no_person_as_motor:
        for i, (pb, pc) in enumerate(persons):
            if i not in used and pc >= a.person_conf:
                b = pb.copy()
                b[3] += 0.4 * (pb[3] - pb[1])  # kéo xuống để ôm cả thân xe
                out.append((tnames["motor"], b))
    lines = []
    for cid, b in out:
        x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"{cid} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    done += 1
print(f"Đã tạo nhãn nháp cho {done} ảnh. Hãy MỞ RA SỬA bằng công cụ gán nhãn trước khi train.")
