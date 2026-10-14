"""Quét ảnh hưởng của độ phân giải đầu vào (imgsz) và frame rate (stride) trong MỘT lệnh.
Mỗi lần chỉ đổi một yếu tố, các yếu tố còn lại giữ ở mức cơ sở.

Chạy:
    python run_sweep.py --video-dir data/videos --config cam1_config.json --weights yolo11m.pt \
        --gt gt.csv --out runs/sweep --imgsz 640 960 1280 --strides 1 2 3 6

stride = lấy 1 khung mỗi N khung. fps hiệu dụng = fps gốc / stride (30 fps -> 15, 10, 5 fps).
Kết quả: runs/sweep/sweep_results.csv (vẽ biểu đồ F1/Recall/FP-giờ/tốc độ theo từng yếu tố).
Nhiễu, mờ, thiếu sáng, nén: dùng make_degraded.py rồi chạy detect_wrong_way.py + evaluate.py.
"""
import argparse
from pathlib import Path

import pandas as pd

from detect_wrong_way import VIDEO_EXT, pick_cfg, run_video
from evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 6])
    ap.add_argument("--base-imgsz", type=int, default=960)
    ap.add_argument("--base-stride", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    vids = [p for p in sorted(Path(a.video_dir).iterdir()) if p.suffix.lower() in VIDEO_EXT]

    jobs = [("imgsz", v) for v in a.imgsz] + [("stride", v) for v in a.strides]
    rows = []
    for factor, val in jobs:
        kw = dict(imgsz=a.base_imgsz, stride=a.base_stride)
        kw[factor] = val
        out = Path(a.out) / f"{factor}_{val}"
        proc_fps = []
        for v in vids:
            _, meta = run_video(v, pick_cfg(a.config, v), model, out, conf=a.conf, save_video=False, **kw)
            proc_fps.append(meta["proc_fps"] or 0)
        res = evaluate(out, a.gt, a.tol, verbose=False)
        rows.append({"factor": factor, "value": val, **kw, **res,
                     "proc_fps_mean": sum(proc_fps) / len(proc_fps)})
        print(f"[{factor}={val}] P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f} "
              f"FP/h={res['fp_per_hour']:.2f}")
    df = pd.DataFrame(rows).round(4)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "sweep_results.csv", index=False)
    print(df[["factor", "value", "TP", "FP", "FN", "precision", "recall", "f1", "fp_per_hour",
              "mean_delay_s", "proc_fps_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
