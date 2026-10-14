"""Bước 2: vẽ làn đường và chiều đi ĐÚNG của từng làn (chạy trên máy có màn hình).

Chạy:
    python 02_draw_roi.py --video data/videos/sun_01.mp4 --out cam1_config.json

Thao tác (cửa sổ OpenCV):
  - Click trái: thêm đỉnh của polygon làn.
  - Enter: chốt polygon (cần >= 3 đỉnh) -> chuyển sang chọn chiều.
  - Click 2 điểm: điểm đầu -> điểm cuối theo CHIỀU ĐI ĐÚNG của làn đó.
  - z: hoàn tác, s: lưu và thoát, q/ESC: thoát không lưu.
Chỉ vẽ vùng có xe (bỏ lan can bê tông ở dưới). Mỗi chiều đường vẽ một làn.
Không có màn hình (Colab/SSH)? Sửa tay file JSON mẫu cam1_config.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--image")
    ap.add_argument("--frame", type=int, default=0, help="chỉ số khung dùng để vẽ")
    ap.add_argument("--out", default="cam1_config.json")
    ap.add_argument("--max-side", type=int, default=1000, help="thu nhỏ khi hiển thị cho vừa màn hình")
    a = ap.parse_args()

    if a.image:
        frame = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Không đọc được khung hình")
    H, W = frame.shape[:2]
    scale = min(1.0, a.max_side / max(H, W))
    base = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()

    lanes = []  # lưu theo toạ độ ảnh gốc
    S = {"mode": "poly", "poly": [], "poly_done": None, "dir": []}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = int(x / scale), int(y / scale)
        if S["mode"] == "poly":
            S["poly"].append([ox, oy])
        else:
            S["dir"].append([ox, oy])
            if len(S["dir"]) == 2:
                (x0, y0), (x1, y1) = S["dir"]
                lanes.append({"name": f"lane{len(lanes) + 1}", "polygon": S["poly_done"],
                              "direction": [x1 - x0, y1 - y0]})
                S.update(mode="poly", poly=[], poly_done=None, dir=[])

    def s(pt):
        return int(pt[0] * scale), int(pt[1] * scale)

    def render():
        img = base.copy()
        for l in lanes:
            pts = np.array([s(p) for p in l["polygon"]], np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            c = pts.mean(0).astype(int)
            d = np.array(l["direction"], float)
            d = d / (np.linalg.norm(d) + 1e-9) * 80
            cv2.arrowedLine(img, tuple(c), (int(c[0] + d[0]), int(c[1] + d[1])), (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(img, l["name"], tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        pts = [s(p) for p in S["poly"]]
        if pts:
            cv2.polylines(img, [np.array(pts, np.int32)], S["mode"] == "dir", (0, 200, 255), 2)
            for p in pts:
                cv2.circle(img, p, 4, (0, 0, 255), -1)
        for p in S["dir"]:
            cv2.circle(img, s(p), 6, (255, 0, 0), -1)
        msg = ("Click dinh polygon lan, Enter de chot" if S["mode"] == "poly"
               else "Click 2 diem: bat dau -> ket thuc theo CHIEU DI DUNG")
        cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    win = "ve lan (Enter=chot, z=undo, s=luu, q=thoat)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    save = False
    while True:
        cv2.imshow(win, render())
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and S["mode"] == "poly" and len(S["poly"]) >= 3:
            S["poly_done"], S["mode"] = S["poly"][:], "dir"
        elif k == ord("z"):
            if S["mode"] == "dir" and S["dir"]:
                S["dir"].pop()
            elif S["mode"] == "dir":
                S["mode"] = "poly"
            elif S["poly"]:
                S["poly"].pop()
            elif lanes:
                lanes.pop()
        elif k == ord("s"):
            save = True
            break
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()

    if save and lanes:
        cfg = {"frame_size": [W, H], "lanes": lanes, "params": {}}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"Đã lưu {len(lanes)} làn -> {a.out}")
    else:
        print("Không lưu.")


if __name__ == "__main__":
    main()
