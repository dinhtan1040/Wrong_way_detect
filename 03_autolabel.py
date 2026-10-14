"""Bước 5a: dán nhãn nháp bằng YOLO, sau đó bạn SỬA TAY bằng labelImg/CVAT/Roboflow.

Chạy (nên dùng model đã train trên VisDrone để nhãn nháp cùng bộ lớp với model sẽ fine-tune tiếp):
    python 03_autolabel.py --frames data/frames --weights runs/detect/train-7/weights/best.pt

Nhãn ghi cạnh ảnh (cùng tên, đuôi .txt, định dạng YOLO) và DÙNG NGUYÊN ID LỚP CỦA MODEL --weights
(VisDrone: 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor;
 COCO: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck). Chỉ ghi lớp phương tiện (chọn theo tên).
classes.txt (tên mọi lớp của model) được tạo để công cụ dán nhãn đọc.

QUY ƯỚC DÁN NHÃN: xe máy + người ngồi trên xe = MỘT khung, nhãn motorcycle/motor. Không vẽ khung người riêng.
Phải dùng cùng --weights khi chạy 04_make_dataset.py.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from classes import vehicle_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--keep-person", action="store_true",
                    help="ghi cả lớp người vào nhãn nháp để bạn xem và sửa tay (xe máy bị nhận là người -> đổi thành motor)")
    a = ap.parse_args()

    model = YOLO(a.weights)
    ids = vehicle_ids(model.names, a.keep_person)
    print("Lớp ghi vào nhãn nháp:", [f"{i}:{model.names[i]}" for i in ids])
    root = Path(a.frames)
    (root / "classes.txt").write_text("\n".join(model.names[i] for i in range(len(model.names))), encoding="utf-8")
    imgs = sorted(root.rglob("*.jpg"))
    print(f"{len(imgs)} ảnh")
    for n, img in enumerate(imgs, 1):
        r = model.predict(str(img), imgsz=a.imgsz, conf=a.conf, classes=ids, verbose=False)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        img.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
        if n % 100 == 0:
            print(f"  {n}/{len(imgs)}")
    print("Xong. Giờ hãy mở thư mục frames bằng công cụ dán nhãn và sửa lại nhãn nháp.")


if __name__ == "__main__":
    main()
