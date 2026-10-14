"""So sánh nhiều detector (M0 COCO / M1 VisDrone / M2 fine-tune ...) ở MỨC SỰ KIỆN trên video của nhóm.

Chạy:
    python run_models.py --models M0_coco=yolo11m.pt M1_visdrone=runs/detect/train-7/weights/best.pt \
        --video-dir data/videos --config configs --gt gt.csv --out runs/models --imgsz 960 --stride 2

Với model COCO nếu xe máy hay bị gán `person`, thêm --include-person-for M0_coco.
Kết quả: <out>/<tên_model>/... và <out>/models_results.csv (TP/FP/FN, precision, recall, F1, FP/giờ, độ trễ).
Mức A (đo detector trên VisDrone) dùng 05_eval_detector.py.
"""
import argparse
from pathlib import Path

import pandas as pd

from detect_wrong_way import VIDEO_EXT, pick_cfg, run_video
from evaluate import evaluate


def parse_models(items):
    out = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--models cần dạng tên=đường_dẫn.pt, nhận được '{it}'")
        k, v = it.split("=", 1)
        out.append((k, v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="runs/models")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--include-person-for", nargs="*", default=[], help="tên model được thêm lớp người")
    ap.add_argument("--save-video", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    vids = [p for p in sorted(Path(a.video_dir).iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not vids:
        raise SystemExit("Không có video trong " + a.video_dir)
    rows = []
    for tag, w in parse_models(a.models):
        model = YOLO(w)
        out = Path(a.out) / tag
        print(f"== {tag} ({w})")
        for v in vids:
            run_video(v, pick_cfg(a.config, v), model, out, imgsz=a.imgsz, conf=a.conf, stride=a.stride,
                      save_video=a.save_video, include_person=tag in a.include_person_for)
        res = evaluate(out, a.gt, a.tol, verbose=False)
        rows.append({"model": tag, "weights": w, "imgsz": a.imgsz, "stride": a.stride, **res})
        print(f"[{tag}] TP={res['TP']} FP={res['FP']} FN={res['FN']} P={res['precision']:.3f} "
              f"R={res['recall']:.3f} F1={res['f1']:.3f} FP/h={res['fp_per_hour']:.2f}")
    df = pd.DataFrame(rows).round(4)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "models_results.csv", index=False)
    print(df[["model", "TP", "FP", "FN", "precision", "recall", "f1", "fp_per_hour", "mean_delay_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
