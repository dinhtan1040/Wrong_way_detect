"""Đánh giá detector theo NHÓM LỚP CHUNG (motorcycle, car, bus, truck, bicycle) trên một bộ dữ liệu bất kỳ.

Vì sao cần: model COCO, model train trên VisDrone và model fine-tune trên dữ liệu nhóm có bộ lớp khác nhau
(ví dụ motorcycle=3 trong COCO nhưng motor=9 trong VisDrone), nên `yolo val` không so sánh trực tiếp được.
Ở đây cả nhãn thật lẫn dự đoán đều được đổi về nhóm chung theo TÊN lớp (classes.py):
    motor/motorcycle -> motorcycle ; car/van -> car ; bus ; truck ; bicycle
Lớp tricycle, awning-tricycle và các lớp người không được đánh giá (dự đoán và nhãn của chúng bị bỏ qua).

Chỉ số: AP50 (101 điểm recall kiểu COCO), Precision / Recall / F1 tại ngưỡng --conf-thr,
và Recall theo chiều cao khung nhãn (nhỏ < 32 px, vừa 32-96 px, lớn >= 96 px, tính trên ảnh gốc).
Số liệu có thể lệch nhẹ so với `yolo val` vì cách gộp lớp.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from classes import GROUPS, group_of

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CHUNK = 8
SIZE_BINS = [("small", 0, 32), ("medium", 32, 96), ("large", 96, 1e9)]


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def list_images(src):
    """src: thư mục ảnh, file .txt liệt kê ảnh, một ảnh, hoặc danh sách các mục trên."""
    items = src if isinstance(src, (list, tuple)) else [src]
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() == ".txt":
            out += [Path(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            out.append(p)
    return out


def label_path(img):
    """.../images/x.jpg -> .../labels/x.txt (quy ước của Ultralytics)."""
    s = str(img)
    for sep in {os.sep, "/"}:
        a, b = f"{sep}images{sep}", f"{sep}labels{sep}"
        if a in s:
            s = b.join(s.rsplit(a, 1))
            break
    return Path(s).with_suffix(".txt")


def read_gt(path, w, h, names):
    """Trả về [(nhóm, [x1, y1, x2, y2])] theo pixel; bỏ nhãn không thuộc nhóm chung."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        g = group_of(names.get(int(float(t[0])), ""))
        if g is None:
            continue
        xc, yc, bw, bh = (float(v) for v in t[1:5])
        out.append((g, [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
    return out


def box_iou_np(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def match_one(pb, pc, gb, iou_thr=0.5):
    """Ghép tham lam theo độ tin cậy giảm dần. Trả về (tp theo thứ tự dự đoán gốc, conf đã ghép của từng nhãn, -1 nếu chưa ghép)."""
    pb, pc, gb = np.asarray(pb, float).reshape(-1, 4), np.asarray(pc, float).reshape(-1), np.asarray(gb, float).reshape(-1, 4)
    tp = np.zeros(len(pb))
    gm = -np.ones(len(gb))
    if len(pb) == 0 or len(gb) == 0:
        return tp, gm
    iou = box_iou_np(pb, gb)
    used = np.zeros(len(gb), bool)
    for i in np.argsort(-pc):
        cand = np.where(~used)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(iou[i, cand])]
        if iou[i, j] >= iou_thr:
            used[j] = True
            tp[i] = 1
            gm[j] = pc[i]
    return tp, gm


def ap_from(conf, tp, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mpre = np.flip(np.maximum.accumulate(np.flip(prec)))  # đường bao precision giảm dần
    x = np.linspace(0, 1, 101)  # 101 điểm recall, kiểu COCO: lấy precision tại recall >= x, vượt recall tối đa thì 0
    idx = np.searchsorted(rec, x, side="left")
    q = np.where(idx < len(mpre), mpre[np.minimum(idx, len(mpre) - 1)], 0.0)
    return float(q.mean())


def eval_model(model, images, gt_names, imgsz=640, conf_thr=0.25, iou_nms=0.6, device=None, verbose=True):
    """Chạy model trên `images`, so với nhãn cạnh ảnh (đã đổi về nhóm chung). Trả về DataFrame theo nhóm."""
    acc = {g: dict(conf=[], tp=[], gt=[]) for g in GROUPS}
    extra = {"device": device} if device not in (None, "") else {}
    def results():
        # chạy theo từng nhóm nhỏ, batch cố định: nếu để cả danh sách một lần thì có thể thành một batch khổng lồ -> hết VRAM
        paths = [str(p) for p in images]
        for i in range(0, len(paths), CHUNK):
            yield from model.predict(source=paths[i:i + CHUNK], batch=CHUNK, imgsz=imgsz, conf=0.001,
                                     iou=iou_nms, max_det=300, verbose=False, **extra)

    for n, r in enumerate(results(), 1):
        h, w = r.orig_shape[:2]
        gts = read_gt(label_path(r.path), w, h, gt_names)
        b = r.boxes
        xyxy = _np(b.xyxy).reshape(-1, 4)
        conf = _np(b.conf).reshape(-1)
        cls = _np(b.cls).reshape(-1).astype(int)
        pg = np.array([group_of(model.names[c]) or "" for c in cls])
        for g in GROUPS:
            pm = pg == g
            gb = np.array([bx for gg, bx in gts if gg == g]).reshape(-1, 4)
            tp, gm = match_one(xyxy[pm], conf[pm], gb)
            acc[g]["conf"] += conf[pm].tolist()
            acc[g]["tp"] += tp.tolist()
            acc[g]["gt"] += [(float(bx[3] - bx[1]), float(m)) for bx, m in zip(gb, gm)]
        if verbose and n % 200 == 0:
            print(f"  {n}/{len(images)} ảnh")
    rows = []
    for g in GROUPS:
        d = acc[g]
        conf, tp = np.array(d["conf"]), np.array(d["tp"])
        n_gt = len(d["gt"])
        sel = conf >= conf_thr
        TP = int(tp[sel].sum()) if len(tp) else 0
        FP = int(sel.sum()) - TP
        FN = n_gt - TP
        P = TP / (TP + FP) if TP + FP else float("nan")
        R = TP / n_gt if n_gt else float("nan")
        F1 = 2 * P * R / (P + R) if P == P and R == R and P + R > 0 else float("nan")
        row = dict(group=g, n_gt=n_gt, AP50=ap_from(conf, tp, n_gt), P=P, R=R, F1=F1, TP=TP, FP=FP, FN=FN)
        gh = np.array([x[0] for x in d["gt"]]) if n_gt else np.array([])
        gm_ = np.array([x[1] for x in d["gt"]]) if n_gt else np.array([])
        for name, lo, hi in SIZE_BINS:
            m = (gh >= lo) & (gh < hi) if n_gt else np.array([], bool)
            row[f"n_{name}"] = int(m.sum())
            row[f"R_{name}"] = float((gm_[m] >= conf_thr).mean()) if m.sum() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def load_split(data, split):
    """Trả về (danh sách ảnh, {id: tên lớp}). `data` là yaml (ví dụ VisDrone.yaml hoặc dataset/data.yaml)."""
    try:
        from ultralytics.data.utils import check_det_dataset
        d = check_det_dataset(data)
        src, names = d.get(split), d["names"]
    except Exception as e:  # không có ultralytics hoặc không đọc được: tự đọc yaml
        p = Path(data)
        if not p.exists():
            raise SystemExit(f"Không đọc được dataset '{data}': {e}")
        import yaml
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        base = Path(d.get("path", p.parent))
        s = d.get(split)
        src = None if s is None else [str(base / x) for x in (s if isinstance(s, list) else [s])]
        names = d["names"]
    if src is None:
        raise SystemExit(f"Dataset không có split '{split}'")
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else {int(k): v for k, v in names.items()}
    return list_images(src), names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="yaml dataset, ví dụ VisDrone.yaml hoặc dataset/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", nargs="+", required=True, help="một hoặc nhiều model (.pt)")
    ap.add_argument("--names", nargs="*", help="tên hiển thị cho từng model (cùng thứ tự --weights)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thr", type=float, default=0.25, help="ngưỡng để tính Precision/Recall")
    ap.add_argument("--max-images", type=int, default=0, help="chỉ đánh giá N ảnh đầu (0 = tất cả)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="eval_detector.csv")
    a = ap.parse_args()

    from ultralytics import YOLO
    images, gt_names = load_split(a.data, a.split)
    if a.max_images:
        images = images[: a.max_images]
    print(f"{len(images)} ảnh ({a.data}, split={a.split}); lớp nhãn: {list(gt_names.values())}")
    tags = a.names or [Path(w).parent.parent.name + "_" + Path(w).stem if "weights" in Path(w).parts else Path(w).stem
                       for w in a.weights]
    if len(tags) != len(a.weights):
        raise SystemExit("--names phải có cùng số phần tử với --weights")

    frames = []
    for tag, w in zip(tags, a.weights):
        print(f"== {tag} ({w})")
        df = eval_model(YOLO(w), images, gt_names, a.imgsz, a.conf_thr, device=a.device)
        df.insert(0, "model", tag)
        frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    res.insert(1, "dataset", f"{Path(a.data).stem}:{a.split}")
    res.to_csv(a.out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    piv = res.pivot(index="model", columns="group", values="AP50")[GROUPS]
    piv["mAP50"] = piv.mean(axis=1, skipna=True)
    print("\n== AP50 theo nhóm lớp chung (imgsz=%d) ==" % a.imgsz)
    print(piv.round(3).to_string())
    m = res[res.group == "motorcycle"].set_index("model")
    print(f"\n== Xe máy: Precision / Recall tại conf>={a.conf_thr}, Recall theo cỡ khung (px chiều cao) ==")
    print(m[["n_gt", "AP50", "P", "R", "F1", "R_small", "R_medium", "R_large", "n_small", "n_medium", "n_large"]]
          .round(3).to_string())
    print(f"\nĐã lưu {a.out}")


if __name__ == "__main__":
    main()