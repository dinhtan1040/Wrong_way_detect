"""Bước 6: đánh giá và so sánh detector (mức A / B / C) theo nhóm lớp chung.

Ví dụ:
  # Mức A: database gốc VisDrone (imgsz nên bằng lúc train, ví dụ 640)
  python 05_eval_detector.py --data VisDrone.yaml --split val --imgsz 640 \
      --weights yolo11m.pt runs/detect/train-7/weights/best.pt --names M0_coco M1_visdrone --out eval_visdrone_val.csv

  # Dữ liệu nhóm tự quay (tập test chia theo video, tạo bằng 04_make_dataset.py)
  python 05_eval_detector.py --data dataset/data.yaml --split test --imgsz 960 \
      --weights yolo11m.pt runs/detect/train-7/weights/best.pt runs/detect/ft/weights/best.pt \
      --names M0_coco M1_visdrone M2_finetuned --out eval_own_test.csv

Chi tiết cách tính xem detector_eval.py.
"""
from detector_eval import main

if __name__ == "__main__":
    main()
