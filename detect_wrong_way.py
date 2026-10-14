"""Phát hiện phương tiện đi ngược chiều:
Detection (YOLO) -> Tracking (BoT-SORT/ByteTrack) -> Trajectory (điểm chạm đất) -> Direction estimation.

Chạy một video hoặc cả thư mục:
    python detect_wrong_way.py --video data/videos --config cam1_config.json \
        --weights yolo11m.pt --out runs/B_pretrained --imgsz 960 \
        --tracker track_static.yaml --conf 0.15

Mỗi video cho ra trong thư mục --out:
    <ten>_events.csv  sự kiện ngược chiều (dùng cho evaluate.py)
    <ten>_tracks.csv  thống kê mọi track
    <ten>_meta.json   thông tin video / tham số / tốc độ xử lý
    <ten>_out.mp4     video có vẽ khung và cảnh báo (tắt bằng --no-video)

Thay đổi so với bản trước:
  * Lọc box ma: box quá to so với khung hình (--max-box-frac) và ngưỡng conf riêng cho lớp lớn (bus/truck).
  * --drop-classes: bỏ hẳn một số lớp theo tên (vd: bus nếu cảnh không có xe buýt).
  * lane_of có lề (lane_margin_frac) để xe đi sát mép làn không bị đứt quỹ đạo.
  * Thêm --tracker, --device (mặc định tự chọn, không còn ép 'cuda'), --no-show để tắt cửa sổ xem trực tiếp (mặc định VẪN hiện như cũ).
  * Thống kê số box bị lọc trong <ten>_meta.json.
  * Nhãn FPS (tốc độ xử lý thực, làm mượt) và thời điểm t trong video, góc trên trái của cửa sổ và video đầu ra (tắt: --no-fps).
  * COASTING: detector hụt thì tracker phát box SUY ĐOÁN (vẽ vàng cam, nhãn có dấu ~) để quỹ đạo/ID không đứt; khung suy đoán
    KHÔNG làm tăng bộ đếm vi phạm, nên không thể tự tạo cảnh báo ngược chiều.
  * --tracking lane (MẶC ĐỊNH): tracker riêng (lane_tracker.py) CHỈ cấp ID cho xe trong vùng làn và ƯỚC LƯỢNG VỊ TRÍ
    THEO HƯỚNG chạy khi xe bị che. --tracking ultra: dùng BoT-SORT/ByteTrack của Ultralytics như trước.
  * NỐI LẠI ID sau khi bị che (link_gap_s, link_radius_frac): track mới xuất hiện đúng chỗ dự đoán của một track vừa mất
    sẽ kế thừa lịch sử/vi phạm của track cũ. Điều kiện báo ngược chiều tính theo THỜI GIAN (min_viol_s), không theo số khung.
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from classes import vehicle_ids
from lane_tracker import LaneTracker, lane_track_stream  # chọn lớp theo TÊN: dùng được cho model COCO lẫn model train trên VisDrone

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_PARAMS = dict(
    window_s=0.5,            # cửa sổ tính hướng (giây) - độc lập với fps
    min_disp_frac=0.3,       # dịch chuyển tối thiểu trong cửa sổ = tỉ lệ x chiều cao bbox
    cos_thresh=-0.5,         # cos(góc giữa hướng xe và chiều chuẩn) < ngưỡng => ngược (-0.5 ~ 120 độ)
    min_viol_s=0.3,          # phải ngược chiều liên tục tối thiểu (giây) mới báo
    max_box_frac=0.40,       # bỏ box có cao/rộng > 40% khung hình (box ma kiểu "17 bus")
    lane_margin_frac=0.03,   # lề quanh polygon làn (tỉ lệ x chiều cao khung) để giữ quỹ đạo xe sát mép
    link_gap_s=3.0,          # xe bị che (biển báo...) tối đa bao lâu vẫn được NỐI LẠI với track cũ (0 = tắt)
    link_radius_frac=1.2,    # bán kính tìm xe tái xuất = tỉ lệ x chiều cao box, quanh vị trí DỰ ĐOÁN
    lost_s=5.0,              # --tracking lane: xe mất dấu tối đa bao lâu vẫn được giữ ID (trôi theo hướng chạy)
    gate_frac=0.8,           # --tracking lane: bán kính ghép = gate_frac x chiều cao box (+ phần nở theo thời gian mất)
    new_conf=0.4,            # --tracking lane: chỉ tạo ID mới khi conf >= ngưỡng này
    vel_alpha=0.5,           # --tracking lane: độ mượt vận tốc
    static_lost_s=10.0,      # --tracking lane: xe ĐỨNG YÊN được nhớ lâu hơn (giây)
    dup_iomin=0.75,           # --tracking lane: hai box chồng nhau >= ngưỡng này (trên box nhỏ hơn) = trùng vật thể
    appearance=True,         # --tracking lane: dùng histogram màu làm "phiếu bầu" khi ghép (tắt: --no-appearance)
    coast_s=1.5,             # --tracking lane: detector hụt thì SUY ĐOÁN vị trí xe đang chạy tối đa N giây (0 = tắt)
    coast_static_s=4.0,      # --tracking lane: xe ĐỨNG YÊN được suy đoán lâu hơn (giây)
)
# Lớp lớn hay sinh box ma trên mặt đường ướt => đòi conf cao hơn --conf chung
CONF_MIN = {"bus": 0.3, "truck": 0.3}


def load_cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_cfg(config, video):
    """--config có thể là một file JSON dùng chung, hoặc một THƯ MỤC chứa <ten_video>.json
    (mỗi video một cấu hình làn); thiếu file riêng thì dùng default.json trong thư mục đó."""
    p = Path(config)
    if not p.is_dir():
        return load_cfg(p)
    f = p / f"{Path(video).stem}.json"
    if not f.exists():
        f = p / "default.json"
    if not f.exists():
        raise SystemExit(f"Không có {Path(video).stem}.json (hoặc default.json) trong {p}")
    return load_cfg(f)


def build_lanes(cfg, W, H):
    sx, sy = W / cfg["frame_size"][0], H / cfg["frame_size"][1]
    lanes = []
    for l in cfg["lanes"]:
        poly = (np.array(l["polygon"], np.float32) * np.array([sx, sy], np.float32)).astype(np.float32)
        d = np.array(l["direction"], np.float64)
        d = d / (np.linalg.norm(d) + 1e-9)
        lanes.append({"name": l["name"], "poly": poly, "dir": d})
    return lanes


def lane_of(pt, lanes, margin=0.0):
    """Làn chứa điểm pt. margin (px) > 0: chấp nhận điểm nằm NGOÀI polygon tối đa margin px.
    Nếu điểm lọt nhiều làn (lề chồng lấn), chọn làn gần nhất."""
    best, best_d = None, -1e9
    for l in lanes:
        d = cv2.pointPolygonTest(l["poly"], (float(pt[0]), float(pt[1])), True)  # >0 trong, <0 ngoài
        if d >= -margin and d > best_d:
            best, best_d = l, d
    return best


def run_video(video, cfg, model, out_dir, imgsz=960, conf=0.15, stride=1, save_video=True,
              params=None, tracker="botsort.yaml", debug=False, include_person=False,
              device=None, show=True, drop_classes=(), extra_classes=(), tracking="lane", show_fps=True):
    """Xử lý một video. stride=N: chỉ xử lý 1 khung mỗi N khung (giảm fps hiệu dụng).
    debug=True: ghi thêm <ten>_debug.csv (mỗi quan sát một dòng) để chạy diagnose.py.
    include_person=True: theo dõi cả lớp `person` (COCO 0), vì từ trên cao YOLO hay gán xe máy + người lái là person.
    drop_classes: tên các lớp bỏ hẳn (vd ("bus",))."""
    p = {**DEFAULT_PARAMS, **cfg.get("params", {}), **(params or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    eff_fps = fps / stride
    win = max(3, round(p["window_s"] * eff_fps))
    need = max(2, round(p["min_viol_s"] * eff_fps))
    lanes = build_lanes(cfg, W, H)
    margin_px = p["lane_margin_frac"] * H
    # Vùng bỏ qua (cột biển báo, vật tĩnh hay bị nhận nhầm): khai báo "ignore_zones" trong JSON cấu hình
    sx_, sy_ = W / cfg["frame_size"][0], H / cfg["frame_size"][1]
    ignore = [(np.array(z, np.float32) * np.array([sx_, sy_], np.float32)).astype(np.float32)
              for z in cfg.get("ignore_zones", [])]

    hist = defaultdict(lambda: deque(maxlen=win))  # tid -> (x, y, h)
    viol = defaultdict(int)
    first_viol, cos_acc = {}, defaultdict(list)
    flagged, tracks = {}, {}
    writer = None
    last_t = 0.0
    dbg_rows = []
    filtered = Counter()  # đếm box bị lọc theo lý do
    link_gap = 0 if tracking == "lane" else round(p["link_gap_s"] * eff_fps)   # số khung tối đa một xe được phép "biến mất" rồi nối lại
    canon = {}                                    # id thô của tracker -> id chuẩn (sau khi nối)
    seen_raw = set()
    last_seen = {}                                # id chuẩn -> (khung, x, y, cao)
    trail = defaultdict(lambda: deque(maxlen=8))  # id chuẩn -> (khung, x, y) để ước lượng vận tốc
    n_linked = 0

    cls_ids = vehicle_ids(model.names, include_person)
    drop_set = {n.lower() for n in drop_classes}
    cls_ids = [i for i in cls_ids if model.names[i].lower() not in drop_set]
    # Lớp bổ sung theo tên (vd pedestrian, people): xe máy + người lái hay bị model gán vào các lớp này
    name2id = {n.lower(): k for k, n in model.names.items()}
    for n in extra_classes:
        k = name2id.get(n.lower())
        if k is None:
            print(f"[cảnh báo] model không có lớp '{n}'. Lớp có sẵn: {list(model.names.values())}")
        elif k not in cls_ids:
            cls_ids.append(k)
    model.predictor = None  # reset tracker giữa các video
    t0 = time.time()
    n_proc = 0
    t_prev, fps_ema = time.perf_counter(), None   # FPS hiển thị = tốc độ xử lý thực (làm mượt)

    track_kw = dict(
        source=str(video),
        stream=True,
        persist=True,
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        iou=0.4,            # NMS khắt khe hơn: xóa box đè nhau > 40%
        agnostic_nms=True,  # xóa box trùng KỂ CẢ KHÁC lớp (car/van/truck)
        classes=cls_ids,
        vid_stride=stride,
        verbose=False,
    )
    if device is not None:
        track_kw["device"] = device
    tracker_obj = None
    if tracking == "lane":
        def accept(x1, y1, x2, y2, c, cf):  # lọc trước khi gán ID: ghost box, vùng bỏ qua, NGOÀI LÀN
            if (y2 - y1) > p["max_box_frac"] * H or (x2 - x1) > p["max_box_frac"] * W:
                filtered["too_big"] += 1
                return False
            if cf < CONF_MIN.get(model.names[c], 0.0):
                filtered["low_conf_large_cls"] += 1
                return False
            ctr = ((x1 + x2) / 2, (y1 + y2) / 2)
            if any(cv2.pointPolygonTest(z, (float(ctr[0]), float(ctr[1])), False) >= 0 for z in ignore):
                filtered["ignore_zone"] += 1
                return False
            if lane_of(((x1 + x2) / 2, y2), lanes, margin_px) is None:
                filtered["outside_lane"] += 1
                return False
            return True

        pred_kw = {k: v for k, v in track_kw.items() if k not in ("persist", "tracker")}
        tracker_obj = LaneTracker(eff_fps=eff_fps, lost_s=p["lost_s"], static_lost_s=p["static_lost_s"],
                                  gate_frac=p["gate_frac"], new_conf=p["new_conf"], vel_alpha=p["vel_alpha"],
                                  dup_iomin=p["dup_iomin"], appearance=bool(p["appearance"]),
                                  coast_s=p["coast_s"], coast_static_s=p["coast_static_s"],
                                  coast_ok=lambda f: (0 <= f[0] < W and 0 <= f[1] < H
                                                      and lane_of(f, lanes, margin_px) is not None))
        stream = lane_track_stream(model, pred_kw, accept, tracker_obj, dup_iomin=p["dup_iomin"])
    else:
        stream = model.track(**track_kw)

    for i, r in enumerate(stream):
        n_proc += 1
        now = time.perf_counter()
        if i >= 2 and now > t_prev:                  # bỏ 2 khung đầu (model còn khởi động)
            inst = 1.0 / (now - t_prev)
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
        t_prev = now
        t = i * stride / fps
        last_t = t
        draw = r.orig_img.copy() if save_video else None
        if save_video:
            if writer is None:
                h_, w_ = draw.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / f"{stem}_out.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (w_, h_))
            for l in lanes:
                cv2.polylines(draw, [l["poly"].astype(np.int32)], True, (0, 200, 0), 2)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            boxes = r.boxes.xyxy.tolist()
            clss = r.boxes.cls.int().tolist()
            confs = r.boxes.conf.tolist()
            preds = getattr(r.boxes, "pred", None) or [False] * len(ids)   # True = box SUY ĐOÁN (detector hụt)
            for raw, (x1, y1, x2, y2), c, cf, pred in zip(ids, boxes, clss, confs, preds):
                # ---- Lọc box ma (box suy đoán đã đi qua bộ lọc từ lúc còn là detection thật) ----
                if not pred:
                    if (y2 - y1) > p["max_box_frac"] * H or (x2 - x1) > p["max_box_frac"] * W:
                        filtered["too_big"] += 1
                        continue
                    if cf < CONF_MIN.get(model.names[c], 0.0):
                        filtered["low_conf_large_cls"] += 1
                        continue
                    ctr = ((x1 + x2) / 2, (y1 + y2) / 2)
                    if any(cv2.pointPolygonTest(z, (float(ctr[0]), float(ctr[1])), False) >= 0 for z in ignore):
                        filtered["ignore_zone"] += 1
                        continue

                foot = ((x1 + x2) / 2, y2)
                tid = canon.get(raw, raw)
                if raw not in seen_raw:  # id mới: thử nối với một xe vừa biến mất (bị che) đúng chỗ dự đoán
                    seen_raw.add(raw)
                    best, best_d = None, 1e9
                    if link_gap > 0:
                        for oid, (fi_, ox, oy, oh) in last_seen.items():
                            gap = i - fi_
                            if gap < 2 or gap > link_gap:
                                continue
                            tr = trail[oid]
                            vx = vy = 0.0
                            if len(tr) >= 2:
                                (g0, ax, ay), (g1, bx, by) = tr[0], tr[-1]
                                span = max(1, g1 - g0)
                                vx, vy = (bx - ax) / span, (by - ay) / span
                            dist = float(np.hypot(foot[0] - (ox + vx * gap), foot[1] - (oy + vy * gap)))
                            if dist <= p["link_radius_frac"] * max(oh, 20.0) and dist < best_d:
                                best, best_d = oid, dist
                    if best is not None and best != tid:
                        canon[raw] = best
                        tid = best
                        n_linked += 1
                last_seen[tid] = (i, foot[0], foot[1], y2 - y1)
                trail[tid].append((i, foot[0], foot[1]))
                st = tracks.setdefault(tid, {"cls": Counter(), "t_first": t, "t_last": t, "n": 0})
                st["t_last"] = t
                if not pred:
                    st["cls"][c] += 1
                    st["n"] += 1

                lane = lane_of(foot, lanes, margin_px)
                d_lane, d_nh, d_disp, d_dmin, d_gate, d_cos = "", 0, "", "", "", ""
                if lane is not None:
                    d_lane = lane["name"]
                    hist[tid].append((foot[0], foot[1], y2 - y1))
                    hh = hist[tid]
                    d_nh = len(hh)
                    if len(hh) >= win:
                        arr = np.asarray(hh, np.float64)
                        k = max(1, len(arr) // 3)
                        d = arr[-k:, :2].mean(0) - arr[:k, :2].mean(0)  # làm mượt đầu/cuối cửa sổ
                        disp = float(np.linalg.norm(d))
                        dmin = max(4.0, p["min_disp_frac"] * float(np.median(arr[:, 2])))
                        d_disp, d_dmin, d_gate = round(disp, 2), round(dmin, 2), int(disp >= dmin)
                        if pred:  # khung SUY ĐOÁN: giữ nguyên bộ đếm vi phạm (không tăng, không reset), chỉ ghi để xem
                            if disp >= dmin:
                                d_cos = round(float(d @ lane["dir"] / disp), 3)
                        elif disp >= dmin:
                            cosv = float(d @ lane["dir"] / disp)
                            d_cos = round(cosv, 3)
                            if cosv < p["cos_thresh"]:
                                viol[tid] += 1
                                first_viol.setdefault(tid, t)
                                cos_acc[tid].append(cosv)
                            else:
                                viol[tid] = 0
                                first_viol.pop(tid, None)
                                cos_acc[tid].clear()
                        else:  # đứng yên / dịch chuyển quá nhỏ: không kết luận
                            viol[tid] = 0
                            first_viol.pop(tid, None)
                            cos_acc[tid].clear()
                        # báo theo THỜI GIAN vi phạm (không theo số khung) => không bị thiệt khi xe bị che giữa chừng
                        if (not pred and tid not in flagged and tid in first_viol and viol[tid] >= 3
                                and t - first_viol[tid] >= p["min_viol_s"]):
                            flagged[tid] = dict(t_start=first_viol[tid], t_det=t, lane=lane["name"],
                                                mean_cos=float(np.mean(cos_acc[tid])))
                if debug:
                    dbg_rows.append([stem, round(t, 3), tid, model.names[c], d_lane, round(foot[0], 1),
                                     round(foot[1], 1), d_nh, win, d_disp, d_dmin, d_gate, d_cos,
                                     viol[tid], need, p["cos_thresh"], int(pred)])

                if save_video:
                    bad = tid in flagged
                    col = (0, 0, 255) if bad else ((0, 200, 255) if pred else (0, 255, 0))  # vàng cam = suy đoán
                    cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), col, 3 if bad else (1 if pred else 2))
                    label = f"{tid} {model.names[c]}" + ("~" if pred else "") + (" WRONG-WAY" if bad else "")
                    cv2.putText(draw, label, (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    pts = np.array([[int(x), int(y)] for x, y, _ in hist[tid]], np.int32)
                    if len(pts) > 1:
                        cv2.polylines(draw, [pts], False, col, 2)
        if save_video:
            if show_fps:                                 # nhãn FPS + thời điểm trong video, góc trên trái
                label = (f"FPS {fps_ema:.1f}" if fps_ema else "FPS --") + f" | t {t:.1f}s"
                sc = max(0.7, draw.shape[1] / 900)
                th = max(2, int(sc * 2))
                pos = (10, int(35 * sc))
                cv2.putText(draw, label, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3)
                cv2.putText(draw, label, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th)
            writer.write(draw)
            if show:
                cv2.imshow("Wrong-Way Detection (VisDrone)", draw)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # nhấn 'q' trên bàn phím để dừng sớm
                    break
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if debug:
        with open(out_dir / f"{stem}_debug.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "t", "track_id", "cls", "lane", "foot_x", "foot_y", "n_hist", "win",
                        "disp", "disp_min", "gate", "cos", "viol", "need", "cos_thresh", "pred"])
            w.writerows(dbg_rows)

    events = []
    with open(out_dir / f"{stem}_events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "cls", "t_start", "t_det", "lane", "mean_cos"])
        for tid, e in sorted(flagged.items(), key=lambda kv: kv[1]["t_det"]):
            cname = model.names[tracks[tid]["cls"].most_common(1)[0][0]]
            row = [stem, tid, cname, round(e["t_start"], 3), round(e["t_det"], 3), e["lane"], round(e["mean_cos"], 3)]
            w.writerow(row)
            events.append(dict(zip(["video", "track_id", "cls", "t_start", "t_det", "lane", "mean_cos"], row)))
    with open(out_dir / f"{stem}_tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "cls", "t_first", "t_last", "n_obs", "flagged"])
        for tid, st in sorted(tracks.items()):
            w.writerow([stem, tid, model.names[st["cls"].most_common(1)[0][0]],
                        round(st["t_first"], 3), round(st["t_last"], 3), st["n"], int(tid in flagged)])

    elapsed = time.time() - t0
    meta = dict(video=stem, fps=fps, stride=stride, eff_fps=eff_fps, imgsz=imgsz, conf=conf, params=p,
                tracker=str(tracker), tracking=tracking,
                tracker_stats=dict(tracker_obj.stats) if tracker_obj else {}, duration_s=(n_total / fps) if n_total > 0 else last_t,
                frames_processed=n_proc, runtime_s=round(elapsed, 2),
                proc_fps=round(n_proc / elapsed, 2) if elapsed else None,
                n_events=len(events), n_linked=n_linked, include_person=include_person, filtered_boxes=dict(filtered),
                classes=[model.names[i] for i in cls_ids])
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return events, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file video hoặc thư mục chứa video")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15,
                    help="ngưỡng conf đưa vào tracker (thấp để tracker còn xe mờ; box ma đã có bộ lọc riêng)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tracker", default="botsort.yaml", help="vd track_static.yaml")
    ap.add_argument("--device", default=None, help="vd 0 / cuda / cpu; mặc định tự chọn")
    ap.add_argument("--no-show", action="store_true", help="tắt cửa sổ xem trực tiếp (dùng khi chạy trên server/Colab)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-fps", action="store_true", help="không vẽ nhãn FPS và thời điểm lên khung hình (cửa sổ và video đầu ra)")
    ap.add_argument("--debug", action="store_true", help="ghi <ten>_debug.csv để chạy diagnose.py")
    ap.add_argument("--include-person", action="store_true",
                    help="theo dõi cả lớp person (xe máy bị nhận nhầm là người). Một xe hai người có thể ra hai cảnh báo")
    ap.add_argument("--drop-classes", nargs="*", default=[],
                    help="bỏ hẳn các lớp theo tên, vd: --drop-classes bus")
    ap.add_argument("--extra-classes", nargs="*", default=[],
                    help="theo dõi thêm các lớp theo tên, vd: --extra-classes pedestrian people")
    ap.add_argument("--cos-thresh", type=float)
    ap.add_argument("--min-viol-s", type=float)
    ap.add_argument("--window-s", type=float)
    ap.add_argument("--min-disp-frac", type=float)
    ap.add_argument("--max-box-frac", type=float)
    ap.add_argument("--lane-margin-frac", type=float)
    ap.add_argument("--link-gap-s", type=float, help="xe bị che tối đa bao lâu vẫn nối lại ID (0 = tắt)")
    ap.add_argument("--link-radius-frac", type=float)
    ap.add_argument("--tracking", choices=["lane", "ultra"], default="lane",
                    help="lane: tracker riêng chỉ gán ID trong làn + ước lượng vị trí theo hướng; ultra: Ultralytics")
    ap.add_argument("--lost-s", type=float, help="--tracking lane: giữ ID xe mất dấu tối đa N giây")
    ap.add_argument("--gate-frac", type=float)
    ap.add_argument("--new-conf", type=float)
    ap.add_argument("--static-lost-s", type=float, help="--tracking lane: nhớ xe đứng yên tối đa N giây")
    ap.add_argument("--dup-iomin", type=float, help="--tracking lane: ngưỡng coi hai box là trùng vật thể")
    ap.add_argument("--coast-s", type=float, help="--tracking lane: detector hụt thì suy đoán vị trí xe chạy tối đa N giây (0 = tắt)")
    ap.add_argument("--coast-static-s", type=float, help="--tracking lane: như trên nhưng cho xe đứng yên")
    ap.add_argument("--no-appearance", action="store_true", help="--tracking lane: tắt histogram màu")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    drop = {n.lower() for n in a.drop_classes}
    shown = [model.names[i] for i in vehicle_ids(model.names, a.include_person)
             if model.names[i].lower() not in drop]
    print("Lớp được theo dõi:", shown)
    src = Path(a.video)
    vids = [src] if src.is_file() else [p for p in sorted(src.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    over = {k: v for k, v in dict(cos_thresh=a.cos_thresh, min_viol_s=a.min_viol_s, window_s=a.window_s,
                                  min_disp_frac=a.min_disp_frac, max_box_frac=a.max_box_frac,
                                  lane_margin_frac=a.lane_margin_frac, link_gap_s=a.link_gap_s,
                                  link_radius_frac=a.link_radius_frac, lost_s=a.lost_s,
                                  gate_frac=a.gate_frac, new_conf=a.new_conf,
                                  static_lost_s=a.static_lost_s, dup_iomin=a.dup_iomin,
                                  coast_s=a.coast_s, coast_static_s=a.coast_static_s,
                                  appearance=False if a.no_appearance else None).items() if v is not None}
    for v in vids:
        ev, meta = run_video(v, pick_cfg(a.config, v), model, a.out, imgsz=a.imgsz, conf=a.conf, stride=a.stride,
                             save_video=not a.no_video, params=over, tracker=a.tracker, debug=a.debug,
                             include_person=a.include_person, device=a.device, show=not a.no_show, tracking=a.tracking, show_fps=not a.no_fps,
                             drop_classes=a.drop_classes, extra_classes=a.extra_classes)
        print(f"{v.name}: {len(ev)} sự kiện ngược chiều, {meta['proc_fps']} khung/s xử lý, "
              f"box bị lọc: {meta['filtered_boxes']}, ID nối lại: {meta['n_linked']}, tracker: {meta['tracker_stats']}")
        for e in ev:
            print(f"   t={e['t_det']:.1f}s id={e['track_id']} {e['cls']} làn={e['lane']} cos={e['mean_cos']}")


if __name__ == "__main__":
    main()