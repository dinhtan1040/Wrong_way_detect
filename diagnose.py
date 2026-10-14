"""Chẩn đoán vì sao một video không ra sự kiện ngược chiều.

Chạy:
    python detect_wrong_way.py --video data/videos/Wrong_way_1.mp4 --config configs --out runs/debug --debug
    python diagnose.py --debug-csv runs/debug/Wrong_way_1_debug.csv

In ra "phễu lọc": có bao nhiêu track qua được từng bước, bước nào loại hết xe ngược chiều,
rồi bảng các track có cos thấp nhất (xe nghi ngược chiều nhất). Dùng cột t để mở video đúng chỗ.
Thêm --track 123 để xem chi tiết một track theo thời gian.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-csv", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--track", type=int, help="in chi tiết một track_id")
    ap.add_argument("--t-from", type=float)
    ap.add_argument("--t-to", type=float, help="chỉ xét các quan sát trong khoảng thời gian này (giây)")
    a = ap.parse_args()

    df = pd.read_csv(a.debug_csv)
    if df.empty:
        print("File debug rỗng: detector KHÔNG phát hiện/tracker KHÔNG gán ID cho xe nào.")
        print("-> thử --imgsz 1280, --conf 0.2, kiểm tra video có đọc được không.")
        return
    if a.t_from is not None:
        df = df[df.t >= a.t_from]
    if a.t_to is not None:
        df = df[df.t <= a.t_to]
    need, win, thr = int(df.need.iloc[0]), int(df.win.iloc[0]), float(df.cos_thresh.iloc[0])

    if a.track is not None:
        pd.set_option("display.width", 220, "display.max_rows", 500)
        print(df[df.track_id == a.track][["t", "lane", "foot_x", "foot_y", "n_hist", "disp", "disp_min",
                                          "gate", "cos", "viol"]].to_string(index=False))
        return

    df["in_lane"] = df.lane.notna() & (df.lane.astype(str) != "")
    df["evaluated"] = df.disp.notna()
    df["gated"] = df.gate == 1
    df["wrong"] = df.cos.notna() & (df.cos < thr)
    g = df.groupby("track_id")
    s = pd.DataFrame({
        "cls": g.cls.agg(lambda x: x.mode().iat[0]),
        "t_first": g.t.min(), "t_last": g.t.max(),
        "n_obs": g.size(),
        "n_in_lane": g.in_lane.sum(),
        "n_eval": g.evaluated.sum(),
        "n_gate": g.gated.sum(),
        "min_cos": g.cos.min(),
        "med_cos": g.cos.median(),
        "n_wrong": g.wrong.sum(),
        "max_viol": g.viol.max(),
    })
    s["flagged"] = s.max_viol >= need

    n = len(s)
    print(f"Tham số dùng: cửa sổ = {win} khung, cần ngược liên tục >= {need} khung, cos_thresh = {thr}")
    print("\n== PHỄU LỌC (số track qua được từng bước) ==")
    print(f"1. Tracker tạo ra              : {n} track")
    print(f"2. Từng nằm trong polygon làn  : {int((s.n_in_lane > 0).sum())}")
    print(f"3. Đủ dài để tính hướng        : {int((s.n_eval > 0).sum())}   (cần >= {win} khung trong làn)")
    print(f"4. Dịch chuyển đủ lớn (gate)   : {int((s.n_gate > 0).sum())}")
    print(f"5. Có lúc cos < {thr}          : {int((s.n_wrong > 0).sum())}")
    print(f"6. Ngược liên tục >= {need} khung: {int(s.flagged.sum())}   <- số sự kiện được báo")

    pd.set_option("display.width", 220, "display.max_columns", 30)
    cand = s[s.n_gate > 0].sort_values("min_cos").head(a.top)
    print(f"\n== {a.top} track có cos thấp nhất (xe nghi ngược chiều nhất) ==")
    if cand.empty:
        print("(không có track nào qua bước dịch chuyển đủ lớn)")
    else:
        print(cand.round(2).to_string())

    print("\n== Gợi ý ==")
    if (s.n_in_lane > 0).sum() == 0:
        print("- Không có xe nào nằm trong polygon: polygon trong configs/<tên_video>.json không phủ vùng xe chạy "
              "(hoặc cấu hình bị vẽ cho video khác). Vẽ lại bằng 02_draw_roi.py.")
    elif (s.n_eval > 0).sum() == 0:
        print(f"- Track trong làn quá ngắn (< {win} khung): tracker đứt quỹ đạo hoặc xe ở trong polygon quá ngắn. "
              "Giảm --window-s, mở rộng polygon, hoặc thử --imgsz 1280 / --conf 0.2.")
    elif (s.n_gate > 0).sum() == 0:
        print("- Mọi xe bị loại vì dịch chuyển quá nhỏ: giảm --min-disp-frac (ví dụ 0.1) hoặc tăng fps xử lý (--stride 1).")
    elif (s.n_wrong > 0).sum() == 0:
        allcos = df.cos.dropna()
        print(f"- Có xe di chuyển nhưng chưa xe nào có cos < {thr}. cos trung vị của mọi quan sát = {allcos.median():.2f}, "
              f"nhỏ nhất = {allcos.min():.2f}.")
        print("  Nếu min_cos của xe ngược chiều thật vẫn dương: chiều đúng (direction) của làn đó trong cấu hình có thể "
              "bị vẽ ngược, hoặc xe ngược chiều nằm ngoài polygon (xem cột lane/foot_x/foot_y bằng --track).")
        print("  Nếu min_cos khoảng -0.3..0: tăng --cos-thresh (ví dụ 0) để nhạy hơn.")
    elif s.flagged.sum() == 0:
        print(f"- Có lúc cos < {thr} nhưng không đủ {need} khung liên tiếp (max_viol cao nhất = {int(s.max_viol.max())}). "
              "Tracker đổi ID giữa chừng hoặc xe ra khỏi polygon quá nhanh: giảm --min-viol-s, hoặc kiểm tra bằng --track <id>.")
    else:
        print("- Hệ thống đã báo sự kiện. Dùng cột t_first/t_last để mở video xem có đúng xe cần tìm không.")
    print("\nXem một track theo thời gian: python diagnose.py --debug-csv <file> --track <track_id>")


if __name__ == "__main__":
    main()
