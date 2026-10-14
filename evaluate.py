"""Đánh giá sự kiện ngược chiều: TP / FP / FN, precision, recall, F1, FP/giờ, độ trễ.

Nhãn thật gt.csv (bạn xem video và ghi tay), mỗi dòng là MỘT xe ngược chiều:
    video,t_start,t_end
    rain_01,12.4,15.0
    night_02,40.0,43.5
(video = tên file không đuôi; t_start/t_end tính bằng giây = khoảng thời gian xe đang đi ngược.)
Video không có xe ngược chiều thì KHÔNG cần dòng nào, nhưng vẫn phải được chạy qua detect_wrong_way.py
để tính FP/giờ.

Quy tắc ghép: một cảnh báo (t_det) là đúng nếu rơi trong [t_start - tol, t_end + tol] và chưa ghép với
nhãn nào khác. Cảnh báo thừa = FP, nhãn không có cảnh báo = FN.

Chạy:
    python evaluate.py --pred runs/B_pretrained --gt gt.csv --tol 2.0
Kết quả được tách theo nhóm điều kiện = phần trước dấu "_" đầu tiên trong tên video (rain, sun, night...).
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def match(gt_rows, preds, tol):
    used, tp, delays = set(), 0, []
    for g in sorted(gt_rows, key=lambda x: x[0]):
        best = None
        for j, td in enumerate(preds):
            if j in used:
                continue
            if g[0] - tol <= td <= g[1] + tol and (best is None or td < preds[best]):
                best = j
        if best is not None:
            used.add(best)
            tp += 1
            delays.append(preds[best] - g[0])
    return tp, len(preds) - len(used), len(gt_rows) - tp, delays


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan")
    return p, r, f


def summarize(d):
    tp, fp, fn = int(d.TP.sum()), int(d.FP.sum()), int(d.FN.sum())
    p, r, f = prf(tp, fp, fn)
    hrs = float(d.hours.sum())
    nd = int(d.n_delay.sum())
    return dict(TP=tp, FP=fp, FN=fn, precision=p, recall=r, f1=f,
                fp_per_hour=fp / hrs if hrs else float("nan"),
                mean_delay_s=float(d.delay_sum.sum()) / nd if nd else float("nan"), hours=hrs)


def evaluate(pred_dir, gt_csv, tol=2.0, verbose=True):
    pred_dir = Path(pred_dir)
    gt = pd.read_csv(gt_csv)
    gt["video"] = gt["video"].map(lambda v: Path(str(v)).stem)
    rows = []
    for mf in sorted(pred_dir.glob("*_meta.json")):
        meta = json.loads(mf.read_text(encoding="utf-8"))
        stem = mf.name[: -len("_meta.json")]
        evf = pred_dir / f"{stem}_events.csv"
        preds = pd.read_csv(evf)["t_det"].tolist() if evf.exists() else []
        g = [(r.t_start, r.t_end) for r in gt[gt.video == stem].itertuples()]
        tp, fp, fn, delays = match(g, preds, tol)
        rows.append(dict(video=stem, group=stem.split("_")[0], TP=tp, FP=fp, FN=fn,
                         hours=meta["duration_s"] / 3600, delay_sum=sum(delays), n_delay=len(delays)))
    if not rows:
        raise SystemExit(f"Không có *_meta.json trong {pred_dir}")
    df = pd.DataFrame(rows)
    overall = summarize(df)
    groups = pd.DataFrame([{"group": k, **summarize(d)} for k, d in df.groupby("group")])
    df.drop(columns=["delay_sum", "n_delay"]).to_csv(pred_dir / "eval_per_video.csv", index=False)
    pd.concat([pd.DataFrame([{"group": "ALL", **overall}]), groups]).to_csv(pred_dir / "eval_summary.csv", index=False)
    if verbose:
        pd.set_option("display.width", 200)
        print("== Theo video ==")
        print(df.drop(columns=["delay_sum", "n_delay"]).round(3).to_string(index=False))
        print("\n== Theo điều kiện ==")
        print(groups.round(3).to_string(index=False))
        print("\n== Tổng ==")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in overall.items()})
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="thư mục kết quả của detect_wrong_way.py")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=2.0, help="dung sai thời gian (giây)")
    a = ap.parse_args()
    evaluate(a.pred, a.gt, a.tol)


if __name__ == "__main__":
    main()
