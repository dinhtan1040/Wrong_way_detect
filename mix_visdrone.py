"""Trộn dữ liệu công khai VisDrone với dữ liệu tự quay khi fine-tune
(giữ kiến thức VisDrone, đúng yêu cầu đề: database công khai + dữ liệu nhóm tự thu thập).

    python mix_visdrone.py --visdrone datasets/VisDrone/VisDrone2019-DET-train --own dataset --n-visdrone 1500 --repeat-own 3

Tạo dataset/train_mix.txt và dataset/data_mix.yaml. Val vẫn là clip val TỰ QUAY (đo đúng miền đích).
VisDrone cần có thư mục images/ và labels/ (định dạng YOLO). Nếu chưa có, chạy một lần
    yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=1 imgsz=320
để Ultralytics tự tải và chuyển nhãn (thường vào thư mục datasets/VisDrone).
"""
import argparse
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--visdrone", required=True, help="thư mục VisDrone2019-DET-train (chứa images/ và labels/)")
ap.add_argument("--own", default="dataset", help="thư mục do split_by_clip.py tạo")
ap.add_argument("--n-visdrone", type=int, default=1500)
ap.add_argument("--repeat-own", type=int, default=3, help="lặp ảnh tự quay k lần để không bị VisDrone lấn át")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

vis, own = Path(a.visdrone), Path(a.own)
if not (vis / "images").is_dir() or not (vis / "labels").is_dir():
    raise SystemExit(f"{vis} thiếu images/ hoặc labels/ (xem hướng dẫn đầu file).")
vis_imgs = sorted(p for p in (vis / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
own_imgs = sorted(p for p in (own / "images" / "train").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not own_imgs:
    raise SystemExit("Không có ảnh train tự quay trong " + str(own / "images" / "train"))
rng = random.Random(a.seed)
vis_pick = rng.sample(vis_imgs, min(a.n_visdrone, len(vis_imgs)))
lines = [p.resolve().as_posix() for p in vis_pick] + [p.resolve().as_posix() for p in own_imgs] * a.repeat_own
rng.shuffle(lines)
(own / "train_mix.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

src_yaml = (own / "data.yaml").read_text(encoding="utf-8").splitlines()
names_block = src_yaml[src_yaml.index("names:"):]
yaml = [f"path: {own.resolve().as_posix()}", "train: train_mix.txt", "val: images/val"] + names_block
(own / "data_mix.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
print(f"train_mix.txt: {len(vis_pick)} ảnh VisDrone + {len(own_imgs)} ảnh tự quay x{a.repeat_own} = {len(lines)} dòng")
print(f"Đã ghi {own / 'data_mix.yaml'}")