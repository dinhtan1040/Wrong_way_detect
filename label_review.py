"""Xem và SỬA nhãn YOLO (cặp .jpg + .txt) ngay trên máy, chỉ cần OpenCV; có thể DỰ ĐOÁN SẴN bằng model.
    python label_review.py dataset_all --target runs/detect/train-7/weights/best.pt
    python label_review.py dataset_all --target runs/detect/train-7/weights/best.pt --predict runs/detect/ft-v6/weights/best.pt

--target : weights chứa danh sách lớp/id của BỘ NHÃN (thường là model gốc train-7).
--predict: model dùng để dự đoán sẵn. Ảnh CHƯA có .txt sẽ được điền sẵn khung dự đoán (tắt: --no-auto).
           Khớp lớp theo TÊN; --alias van=car tricycle=motor để gộp lớp; lớp không khớp bị bỏ.

Chuột:  kéo chuột trái = vẽ khung mới (lớp đang chọn)
        click trái vào khung = ĐỔI lớp khung đó thành lớp đang chọn
        click phải vào khung = XÓA khung
Phím:   1=car  2=truck  3=bus  4=motor  5=van   (chọn lớp; đổi danh sách bằng --classes car motor van ...)
        d / mũi tên phải = ảnh sau (tự lưu)     a / mũi tên trái = ảnh trước
        u = hoàn tác khung cuối    x = xóa hết khung (ảnh nền, không có xe)
        v = bật/tắt LỚP PHỦ dự đoán: trắng mảnh = model và nhãn khớp; TÍM = model thấy mà nhãn KHÔNG có (nghi thiếu nhãn)
        p = THÊM các khung dự đoán còn thiếu (những khung tím)     r = THAY toàn bộ bằng dự đoán của model
        q = lưu và thoát  (lần sau mở lại sẽ nhảy tới ảnh chưa duyệt đầu tiên)
"""
import argparse
from pathlib import Path

import cv2

DEFAULT_CLASSES = ["car", "truck", "bus", "motor", "van"]   # phím 1..9 theo thứ tự này
COLORS = {"car": (0, 200, 0), "truck": (0, 165, 255), "bus": (255, 128, 0), "motor": (0, 0, 255),
          "van": (0, 255, 255), "tricycle": (255, 255, 0), "awning-tricycle": (200, 200, 0),
          "bicycle": (180, 105, 255)}


def make_hotkeys(classes, id_of):
    """Gán phím 1..9 cho các lớp theo thứ tự; báo lỗi nếu lớp không có trong bộ nhãn."""
    bad = [c for c in classes if c not in id_of]
    if bad:
        raise SystemExit(f"Bộ nhãn không có lớp {bad}. Lớp có: {list(id_of)}")
    return {ord(str(i + 1)): c for i, c in enumerate(classes[:9])}


def load_boxes(txt, W, H):
    boxes = []
    if txt.exists():
        for ln in txt.read_text().splitlines():
            v = ln.split()
            if len(v) == 5:
                c, x, y, w, h = int(v[0]), *map(float, v[1:])
                boxes.append([c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return boxes


def save_boxes(txt, boxes, W, H):
    out = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))
        if x2 - x1 >= 3 and y2 - y1 >= 3:
            out.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    txt.write_text("\n".join(out) + ("\n" if out else ""))


def box_iou(a, b):
    ax1, ax2 = sorted((a[0], a[2]))
    ay1, ay2 = sorted((a[1], a[3]))
    bx1, bx2 = sorted((b[0], b[2]))
    by1, by2 = sorted((b[1], b[3]))
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def build_map(pred_names, id_of, alias=None):
    """id lớp của model dự đoán -> id lớp của BỘ NHÃN, khớp theo TÊN (alias áp dụng trước). Lớp không khớp bị bỏ."""
    alias = {k.lower(): v.lower() for k, v in (alias or {}).items()}
    m = {}
    for k, n in pred_names.items():
        n = alias.get(n.lower(), n.lower())
        if n in id_of:
            m[k] = id_of[n]
    return m


def predict_boxes(model, img, cmap, conf, imgsz):
    """Trả về [[cid, x1, y1, x2, y2, conf], ...] theo id của bộ nhãn."""
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5, agnostic_nms=True, verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes.xyxy):
        for (x1, y1, x2, y2), k, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            if k in cmap:
                out.append([cmap[k], x1, y1, x2, y2, c])
    return out


def missing_preds(boxes, preds, thr=0.4):
    """Khung dự đoán KHÔNG trùng khung nhãn nào (IoU < thr) = nghi thiếu nhãn."""
    return [p for p in preds if all(box_iou(p[1:5], b[1:5]) < thr for b in boxes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--target", required=True, help="weights model gốc (lấy id lớp của bộ nhãn)")
    ap.add_argument("--predict", help="weights model dự đoán sẵn (vd runs/detect/ft-v6/weights/best.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng conf của dự đoán")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--no-auto", action="store_true", help="không tự điền dự đoán cho ảnh chưa có nhãn (vẫn dùng phím v/p/r)")
    ap.add_argument("--alias", nargs="*", default=[], help="gộp lớp khi dự đoán, vd: van=car tricycle=motor")
    ap.add_argument("--height", type=int, default=950, help="chiều cao cửa sổ hiển thị")
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                    help="lớp gán phím 1..9 theo thứ tự (mặc định: car truck bus motor van)")
    a = ap.parse_args()

    from ultralytics import YOLO
    names = {k: n.lower() for k, n in YOLO(a.target).names.items()}
    id_of = {n: k for k, n in names.items()}
    hotkeys = make_hotkeys([c.lower() for c in a.classes], id_of)

    pm, cmap = None, {}
    if a.predict:
        pm = YOLO(a.predict)
        alias = dict(x.split("=", 1) for x in a.alias if "=" in x)
        cmap = build_map(pm.names, id_of, alias)
        dropped = [n for k, n in pm.names.items() if k not in cmap]
        print("Ánh xạ lớp dự đoán -> bộ nhãn:", {pm.names[k]: names[v] for k, v in cmap.items()})
        if dropped:
            print("Lớp dự đoán BỊ BỎ (không có trong bộ nhãn, dùng --alias nếu muốn gộp):", dropped)
        if not cmap:
            raise SystemExit("Không lớp nào của model dự đoán khớp với bộ nhãn.")

    folder = Path(a.folder)
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        raise SystemExit("Không có ảnh trong " + str(folder))
    rv_file = folder / "reviewed.txt"
    reviewed = set(rv_file.read_text().split()) if rv_file.exists() else set()
    idx = next((i for i, p in enumerate(imgs) if p.stem not in reviewed), 0)

    st = dict(cur="motor" if "motor" in hotkeys.values() else next(iter(hotkeys.values())), boxes=[], img=None, W=0, H=0, s=1.0, start=None, now=None,
              preds=None, show=False, auto=False)

    def get_preds():
        if pm is None:
            return []
        if st["preds"] is None:
            st["preds"] = predict_boxes(pm, st["img"], cmap, a.conf, a.imgsz)
        return st["preds"]

    def open_img(i):
        im = cv2.imread(str(imgs[i]))
        st["img"], st["H"], st["W"] = im, im.shape[0], im.shape[1]
        st["s"] = a.height / st["H"]
        txt = imgs[i].with_suffix(".txt")
        st["boxes"] = load_boxes(txt, st["W"], st["H"])
        st["preds"], st["auto"] = None, False
        if pm is not None and not a.no_auto and not txt.exists():   # ảnh CHƯA có nhãn: điền sẵn dự đoán
            st["boxes"] = [p[:5] for p in get_preds()]
            st["auto"] = True

    def commit(i):
        save_boxes(imgs[i].with_suffix(".txt"), st["boxes"], st["W"], st["H"])
        reviewed.add(imgs[i].stem)
        rv_file.write_text("\n".join(sorted(reviewed)))

    def box_at(x, y):  # khung nhỏ nhất chứa điểm (ảnh gốc)
        hit = [b for b in st["boxes"] if min(b[1], b[3]) <= x <= max(b[1], b[3]) and min(b[2], b[4]) <= y <= max(b[2], b[4])]
        return min(hit, key=lambda b: abs(b[3] - b[1]) * abs(b[4] - b[2])) if hit else None

    def on_mouse(ev, mx, my, flags, _):
        x, y = mx / st["s"], my / st["s"]
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["start"], st["now"] = (x, y), (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["start"]:
            st["now"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["start"]:
            x0, y0 = st["start"]
            st["start"] = None
            if abs(x - x0) * st["s"] > 6 and abs(y - y0) * st["s"] > 6:
                st["boxes"].append([id_of[st["cur"]], x0, y0, x, y])
                st["auto"] = False
            else:
                b = box_at(x, y)
                if b:
                    b[0] = id_of[st["cur"]]
        elif ev == cv2.EVENT_RBUTTONDOWN:
            b = box_at(x, y)
            if b:
                st["boxes"].remove(b)

    win = "label_review"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    open_img(idx)
    while True:
        view = cv2.resize(st["img"], None, fx=st["s"], fy=st["s"])
        if st["show"] and pm is not None:                       # lớp phủ dự đoán
            for p in get_preds():
                matched = any(box_iou(p[1:5], b[1:5]) >= 0.4 for b in st["boxes"])
                q1, q2 = (int(p[1] * st["s"]), int(p[2] * st["s"])), (int(p[3] * st["s"]), int(p[4] * st["s"]))
                col = (255, 255, 255) if matched else (255, 0, 255)
                cv2.rectangle(view, q1, q2, col, 1 if matched else 2)
                if not matched:
                    cv2.putText(view, f"? {names.get(p[0], p[0])} {p[5]:.2f}", (q1[0], q2[1] + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        for c, x1, y1, x2, y2 in st["boxes"]:
            n = names.get(c, str(c))
            col = COLORS.get(n, (200, 200, 200))
            p1, p2 = (int(x1 * st["s"]), int(y1 * st["s"])), (int(x2 * st["s"]), int(y2 * st["s"]))
            cv2.rectangle(view, p1, p2, col, 2)
            cv2.putText(view, n, (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if st["start"]:
            cv2.rectangle(view, (int(st["start"][0] * st["s"]), int(st["start"][1] * st["s"])),
                          (int(st["now"][0] * st["s"]), int(st["now"][1] * st["s"])), COLORS[st["cur"]], 1)
        extra = ""
        if pm is not None:
            extra = " | AUTO-PREDICTED" if st["auto"] else ""
            if st["preds"] is not None:
                extra += f" | model: {len(st['preds'])} ({len(missing_preds(st['boxes'], st['preds']))} missing)"
        bar = f"{idx + 1}/{len(imgs)} {imgs[idx].name} | class: {st['cur']} | boxes: {len(st['boxes'])}" \
              f"{' | reviewed' if imgs[idx].stem in reviewed else ''}{extra}"
        legend = "keys: " + "  ".join(f"{chr(k)}={n}" for k, n in hotkeys.items()) + "   (v overlay, p add, r replace)"
        cv2.putText(view, bar, (6, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(view, legend, (6, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(st["cur"], (200, 200, 200)), 1)
        cv2.imshow(win, view)
        k = cv2.waitKeyEx(20)
        if k == -1:
            continue
        if k in hotkeys:
            st["cur"] = hotkeys[k]
        elif k in (ord("d"), 2555904):          # d / mũi tên phải
            commit(idx)
            if idx < len(imgs) - 1:
                idx += 1
                open_img(idx)
        elif k in (ord("a"), 2424832):          # a / mũi tên trái
            commit(idx)
            if idx > 0:
                idx -= 1
                open_img(idx)
        elif k == ord("u") and st["boxes"]:
            st["boxes"].pop()
        elif k == ord("x"):
            st["boxes"].clear()
            st["auto"] = False
        elif k == ord("v") and pm is not None:
            st["show"] = not st["show"]
        elif k == ord("p") and pm is not None:
            st["boxes"] += [m[:5] for m in missing_preds(st["boxes"], get_preds())]
            st["auto"] = False
        elif k == ord("r") and pm is not None:
            st["boxes"] = [m[:5] for m in get_preds()]
        elif k == ord("q"):
            commit(idx)
            break
    cv2.destroyAllWindows()
    print(f"Đã duyệt {len(reviewed)}/{len(imgs)} ảnh.")


if __name__ == "__main__":
    main()