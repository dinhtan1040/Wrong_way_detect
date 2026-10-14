"""Tracker nhẹ cho camera CỐ ĐỊNH, dùng chung với detect_wrong_way.py (--tracking lane).  [bản 3 = gộp]

Gộp "bản 2" (xác nhận track, lọc box ma, xử lý box dính/tách, nối ID theo màu) với phần
NHỚ + SUY ĐOÁN VỊ TRÍ (coasting) của bản trước. Giao diện giữ nguyên bản mới nhất của detect_wrong_way.py.

  1. CHỈ cấp ID cho detection trong làn (bộ lọc `accept` do detect_wrong_way.py truyền vào).
  2. XÁC NHẬN TRACK: box chưa ghép chỉ tạo track TẠM; ID số chỉ cấp khi đủ `min_hits` khung (và có ít nhất một khung
     conf >= new_conf) hoặc tồn tại bền `persist_hits` khung => box ma một-hai khung không sinh ID.
  3. CHỐNG TRÙNG VẬT THỂ: (a) mức detection: box nhỏ nằm trong box lớn (IoMin >= dup_iomin) bị bỏ; (b) box chưa ghép đè lên
     box đã ghép (IoU > dup_iou) bị bỏ; (c) mỗi khung, hai track đè nhau IoU > dedupe_iou thì xóa track trẻ hơn.
  4. XE ĐỨNG YÊN: giữ lâu hơn (static_lost_s), vị trí dự đoán không "trôi", ghép theo mức chồng lấp với box cũ.
  5. BOX DÍNH/TÁCH: box đổi kích thước đột ngột (> size_jump lần) chưa được tin ngay, giữ quỹ đạo theo vận tốc cũ.
  6. NỐI ID THEO MÀU: track mới xác nhận gần track vừa mất (cùng màu, cùng hướng) kế thừa ID; khác màu rõ thì không.
  7. SUY ĐOÁN VỊ TRÍ (coasting): khi detector hụt, xe đã theo dõi chắc (>= 3 khớp) vẫn được PHÁT RA box dự đoán
     (vị trí cũ + vận tốc x số khung mất) với pred=True, tối đa coast_s (xe chạy) / coast_static_s (xe đứng yên); dừng khi
     vị trí dự đoán ra khỏi làn/khung (coast_ok). Box pred KHÔNG được dùng để tăng bộ đếm vi phạm ở detect_wrong_way.py.
  8. VẬT MA ĐỨNG YÊN: đứng yên nhiều giây mà conf trung bình thấp (nước đọng, bóng, phản chiếu) thì không phát ra.
"""
from collections import Counter

import cv2
import numpy as np


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _iomin(a, b):
    m = min(_area(a), _area(b))
    return _inter(a, b) / m if m > 0 else 0.0


def _ratio(a, b):
    mx = max(_area(a), _area(b))
    return min(_area(a), _area(b)) / mx if mx > 0 else 0.0


def _feat(img, box):
    """Histogram màu BGR 4x4x4 (chuẩn hoá) của phần giữa box; None nếu không có ảnh hoặc box quá nhỏ."""
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 + 0.15 * w)), int(min(W, x2 - 0.15 * w))
    y1, y2 = int(max(0, y1 + 0.15 * h)), int(min(H, y2 - 0.15 * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hist = cv2.calcHist([img[y1:y2, x1:x2]], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).flatten().astype(np.float32)
    s = float(hist.sum())
    return hist / s if s > 0 else None


def _app_dist(a, b):
    """0 = giống hệt, 1 = khác hoàn toàn. Thiếu thông tin thì trả 0 (không có bằng chứng để loại)."""
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class _Track:
    def __init__(self, uid, i, box, foot, cls, conf, feat):
        self.uid, self.cid = uid, None      # cid = ID hiển thị, chỉ có khi đã xác nhận
        self.last_i, self.box = i, box
        self.foot = np.asarray(foot, float)
        self.vel = np.zeros(2)               # px / khung (đã làm mượt)
        self.h = box[3] - box[1]
        self.hits = 1
        self.cls, self.conf, self.max_conf = cls, conf, conf
        self.feat = feat
        self.born_i = i
        self.anchor = self.foot.copy()       # vị trí lúc xuất hiện (để biết xe có thực sự di chuyển không)
        self.max_move = 0.0
        self.conf_sum, self.n_obs = conf, 1  # tổng conf và số lần THỰC SỰ được detect
        self.cls_w = {cls: conf}             # phiếu bầu lớp, trọng số bằng conf
        self.jump_n = 0                      # số khung liên tiếp box đổi kích thước đột ngột
        self.pred_now = False                # khung này box là SUY ĐOÁN (không phải detection thật)

    @property
    def cls_vote(self):
        return max(self.cls_w, key=self.cls_w.get)


class LaneTracker:
    def __init__(self, eff_fps=30.0, lost_s=3.0, static_lost_s=10.0, gate_frac=0.8, new_conf=0.3, vel_alpha=0.5,
                 err_frac=0.25, static_speed_frac=0.15, min_hits=3, persist_hits=8, tent_lost=3,
                 dup_iou=0.45, dup_iomin=0.75, dedupe_iou=0.6, relink_frac=1.0, app_thr=0.6,
                 size_jump=1.6, size_persist_s=0.25, ghost_s=4.0, ghost_move_frac=0.3, ghost_conf=0.45,
                 appearance=True, coast_s=1.5, coast_static_s=4.0, coast_ok=None):
        self.eff_fps = eff_fps
        self.max_lost = max(3, round(lost_s * eff_fps))               # xe đang chạy: số khung tối đa được "biến mất"
        self.max_lost_static = max(self.max_lost, round(static_lost_s * eff_fps))  # xe đứng yên: nhớ lâu hơn
        self.static_speed_frac = static_speed_frac  # đứng yên nếu tốc độ < tỉ lệ x chiều cao box mỗi giây
        self.gate_frac, self.new_conf, self.alpha, self.err_frac = gate_frac, new_conf, vel_alpha, err_frac
        self.min_hits, self.persist_hits, self.tent_lost = min_hits, persist_hits, tent_lost
        self.dup_iou, self.dup_iomin, self.dedupe_iou = dup_iou, dup_iomin, dedupe_iou
        self.relink_frac, self.app_thr, self.appearance = relink_frac, app_thr, appearance
        self.size_jump = size_jump
        self.size_persist = max(2, round(size_persist_s * eff_fps))
        self.ghost_s, self.ghost_move_frac, self.ghost_conf = ghost_s, ghost_move_frac, ghost_conf
        self.max_coast = round(coast_s * eff_fps)                     # 0 = tắt suy đoán
        self.max_coast_static = round(coast_static_s * eff_fps)
        self.coast_ok = coast_ok                                      # f(foot)->bool: vị trí dự đoán còn hợp lệ (trong làn/khung)
        self.tracks = {}                # uid -> _Track (cả track tạm lẫn đã xác nhận)
        self.next_uid = 1
        self.next_id = 1                # chỉ tăng khi XÁC NHẬN
        self.stats = Counter()

    # ---- tiện ích ----
    def _static(self, tr):
        return tr.hits >= 3 and float(np.linalg.norm(tr.vel)) * self.eff_fps < self.static_speed_frac * max(tr.h, 1.0)

    def _limit(self, tr):
        if tr.cid is None:
            return self.tent_lost
        return self.max_lost_static if self._static(tr) else self.max_lost

    def _is_dup(self, b, taken):
        for t in taken:
            if _iou(b, t) > self.dup_iou:
                return True
            if _iomin(b, t) >= self.dup_iomin and _ratio(b, t) >= 0.3:   # box nhỏ nằm trong box lớn
                return True
        return False

    def _confirm(self, tr, i, matched_uids):
        """Track tạm đủ điều kiện -> cấp ID, ưu tiên KẾ THỪA ID của một track đã mất ở gần, cùng màu, cùng hướng."""
        if not ((tr.hits >= self.min_hits and tr.max_conf >= self.new_conf) or tr.hits >= self.persist_hits):
            return False
        best, best_s = None, 1e18
        for uid, o in self.tracks.items():
            if o is tr or o.cid is None or uid in matched_uids:
                continue
            g = i - o.last_i
            if g < 1:
                continue
            st = self._static(o)
            ref = o.foot if st else o.foot + o.vel * g
            h = max(o.h, tr.h)
            dist = float(np.linalg.norm(tr.foot - ref))
            if dist > self.relink_frac * h + (0 if st else self.err_frac * float(np.linalg.norm(o.vel)) * g):
                continue
            ov = _iou(o.box, tr.box)
            if not st and np.linalg.norm(o.vel) > 0.3 and np.linalg.norm(tr.vel) > 0.3:
                if float(o.vel @ tr.vel) / (np.linalg.norm(o.vel) * np.linalg.norm(tr.vel)) < 0.3:
                    continue
            ad = _app_dist(o.feat, tr.feat)
            if ad > self.app_thr + (0.15 if ov >= 0.5 else 0.0):
                continue
            s = dist / max(h, 1.0) + ad
            if s < best_s:
                best, best_s = o, s
        if best is not None:
            tr.cid = best.cid
            tr.hits += best.hits
            tr.born_i, tr.anchor = min(tr.born_i, best.born_i), best.anchor
            tr.max_move = max(tr.max_move, best.max_move)
            tr.conf_sum += best.conf_sum
            tr.n_obs += best.n_obs
            for c, w in best.cls_w.items():
                tr.cls_w[c] = tr.cls_w.get(c, 0.0) + w
            if best.hits >= 3 and np.linalg.norm(tr.vel) < 1e-6:
                tr.vel = best.vel.copy()
            del self.tracks[best.uid]
            self.stats["inherited"] += 1
        else:
            tr.cid = self.next_id
            self.next_id += 1
        return True

    # ---- cập nhật mỗi khung ----
    def update(self, i, dets, img=None):
        """dets: [(x1, y1, x2, y2, cls, conf)] ĐÃ lọc trong làn.
        Trả về [(id, x1, y1, x2, y2, cls, conf, pred)] cho track đã xác nhận (pred=True: box suy đoán)."""
        if not self.appearance:
            img = None
        for uid in [k for k, t in self.tracks.items() if i - t.last_i > self._limit(t)]:
            del self.tracks[uid]
        feet = [np.array([(d[0] + d[2]) / 2, d[3]]) for d in dets]
        dfeat = {}

        def get_feat(j):
            if j not in dfeat:
                dfeat[j] = _feat(img, dets[j][:4])
            return dfeat[j]

        cand = []
        for uid, tr in self.tracks.items():
            g = max(1, i - tr.last_i)
            st = self._static(tr)
            vel = np.zeros(2) if st else tr.vel
            pred = tr.foot + vel * g                          # vị trí DỰ ĐOÁN theo hướng chạy của chính xe
            speed = float(np.linalg.norm(vel))
            pbox = (tr.box[0] + pred[0] - tr.foot[0], tr.box[1] + pred[1] - tr.foot[1],
                    tr.box[2] + pred[0] - tr.foot[0], tr.box[3] + pred[1] - tr.foot[1])
            for j, d in enumerate(dets):
                h = max(tr.h, d[3] - d[1])
                dist = float(np.linalg.norm(feet[j] - pred))
                gate = self.gate_frac * h + self.err_frac * speed * g
                iou_last = _iou(tr.box, d[:4])
                if dist > gate and iou_last < 0.5:
                    continue
                if g > 1 and not st and tr.hits >= 3 and speed > 0.3 and iou_last < 0.3:  # tái xuất: hướng nối khớp hướng chạy
                    v = feet[j] - tr.foot
                    nv = float(np.linalg.norm(v))
                    if nv > 0.3 * h and float(v @ tr.vel) / (nv * speed) < 0.3:
                        continue
                cost = dist / gate
                if g > 3 and tr.cid is not None:              # nối lại sau khi mất dấu: kiểm tra màu
                    ad = _app_dist(tr.feat, get_feat(j))
                    if ad > self.app_thr + (0.15 if iou_last >= 0.5 else 0.0):
                        continue
                    cost = 0.75 * cost + 0.25 * ad
                if max(iou_last, _iou(pbox, d[:4])) > 0.3:
                    cost *= 0.5
                if tr.cid is None:
                    cost += 0.05                              # ưu tiên track đã xác nhận khi hoà
                cand.append((cost, uid, j))
        cand.sort()
        used_t, used_d = set(), set()
        for cost, uid, j in cand:
            if uid in used_t or j in used_d:
                continue
            used_t.add(uid)
            used_d.add(j)
            tr, d = self.tracks[uid], dets[j]
            g = max(1, i - tr.last_i)
            bw, bh = max(tr.box[2] - tr.box[0], 1.0), max(tr.box[3] - tr.box[1], 1.0)
            rw, rh = (d[2] - d[0]) / bw, (d[3] - d[1]) / bh
            jump = (tr.cid is not None and tr.hits >= 5 and g <= 3
                    and max(rw, 1 / max(rw, 1e-6), rh, 1 / max(rh, 1e-6)) > self.size_jump)
            tr.jump_n = tr.jump_n + 1 if jump else 0
            if jump and tr.jump_n <= self.size_persist:
                # box đổi kích thước đột ngột (dính với xe bên cạnh, hoặc bị tách đôi): chưa tin, giữ quỹ đạo theo vận tốc cũ
                dx, dy = tr.vel * g
                tr.box = (tr.box[0] + dx, tr.box[1] + dy, tr.box[2] + dx, tr.box[3] + dy)
                tr.foot = tr.foot + tr.vel * g
                tr.last_i = i
                tr.hits += 1
                tr.pred_now = True
                self.stats["size_jump"] += 1
                continue
            if g > 1:
                self.stats["relinked"] += 1
            inst = (feet[j] - tr.foot) / g
            tr.vel = inst if tr.hits == 1 else self.alpha * inst + (1 - self.alpha) * tr.vel
            tr.foot, tr.box, tr.last_i = feet[j], d[:4], i
            tr.h = 0.7 * tr.h + 0.3 * (d[3] - d[1])
            tr.hits += 1
            tr.pred_now = False
            tr.cls, tr.conf, tr.max_conf = d[4], d[5], max(tr.max_conf, d[5])
            tr.conf_sum += d[5]
            tr.n_obs += 1
            tr.cls_w[d[4]] = tr.cls_w.get(d[4], 0.0) + d[5]
            tr.max_move = max(tr.max_move, float(np.linalg.norm(tr.foot - tr.anchor)))
            f = get_feat(j) if img is not None else None
            if f is not None:
                tr.feat = f if tr.feat is None else 0.9 * tr.feat + 0.1 * f

        # box chưa ghép: bỏ nếu là box trùng của xe đã có, ngược lại tạo track TẠM
        taken = [dets[j][:4] for j in used_d]
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            if self._is_dup(d[:4], taken):
                self.stats["dup_new"] += 1
                continue
            tr = _Track(self.next_uid, i, d[:4], feet[j], d[4], d[5], get_feat(j) if img is not None else None)
            self.next_uid += 1
            self.tracks[tr.uid] = tr
            used_t.add(tr.uid)
            used_d.add(j)
            taken.append(d[:4])
        return self._coast(i, self._emit(i, used_t))

    def _is_ghost(self, t, i):
        """Vật đứng yên suốt nhiều giây mà conf trung bình thấp (nước đọng, bóng, phản chiếu) -> không phát ra.
        Xe đứng thật (kẹt xe, đèn đỏ) có conf cao nên vẫn được giữ; vật ma mà dịch chuyển thì tự được phát lại."""
        age_s = (i - t.born_i + 1) / self.eff_fps
        return (age_s >= self.ghost_s and t.max_move <= self.ghost_move_frac * max(t.h, 1.0)
                and t.conf_sum / max(1, t.n_obs) < self.ghost_conf)

    def _emit(self, i, active_uids):
        act = [self.tracks[u] for u in active_uids if u in self.tracks and self.tracks[u].last_i == i]
        act.sort(key=lambda t: (t.cid is None, -t.hits))
        keep = []
        for t in act:
            if any(_iou(t.box, k.box) > self.dedupe_iou for k in keep):   # hai track đè nhau = cùng một xe: xóa track trẻ
                del self.tracks[t.uid]
                self.stats["dedupe_del"] += 1
                continue
            keep.append(t)
        active = {t.uid for t in keep}
        res = []
        for t in keep:
            if t.cid is None and not self._confirm(t, i, active):
                continue
            if self._is_ghost(t, i):
                self.stats["ghost_hidden"] += 1
                continue
            res.append((t.cid, *t.box, t.cls_vote, t.conf, t.pred_now))
        return res

    def _coast(self, i, res):
        """Detector hụt khung này: phát box SUY ĐOÁN cho các xe đã theo dõi chắc."""
        if self.max_coast <= 0:
            return res
        for tr in list(self.tracks.values()):
            if tr.cid is None or tr.last_i >= i or tr.hits < 3:
                continue
            g = i - tr.last_i
            st = self._static(tr)
            if g > (self.max_coast_static if st else self.max_coast) or self._is_ghost(tr, i):
                continue
            dx, dy = (0.0, 0.0) if st else (tr.vel[0] * g, tr.vel[1] * g)
            if self.coast_ok is not None and not self.coast_ok((tr.foot[0] + dx, tr.foot[1] + dy)):
                continue
            pb = (tr.box[0] + dx, tr.box[1] + dy, tr.box[2] + dx, tr.box[3] + dy)
            if any(_iou(pb, o[1:5]) > self.dedupe_iou for o in res):
                continue
            res.append((tr.cid, *pb, tr.cls_vote, tr.conf, True))
            self.stats["coast_frames"] += 1
        return res


def _dedupe(dets, thr, stats):
    """Bỏ box trùng vật thể ở mức detection (kể cả khác lớp): box nhỏ nằm trong box lớn, giữ box conf cao hơn."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[5]):
        if any(_iomin(d[:4], k[:4]) >= thr and _ratio(d[:4], k[:4]) >= 0.3 for k in keep):
            stats["dup_det"] += 1
            continue
        keep.append(d)
    return keep


# ---- khớp giao diện với kết quả của model.track() để detect_wrong_way.py dùng chung vòng lặp ----
class _L(list):
    def int(self):
        return self

    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, out):
        self.id = _L(o[0] for o in out)
        self.xyxy = _L([o[1], o[2], o[3], o[4]] for o in out)
        self.cls = _L(o[5] for o in out)
        self.conf = _L(o[6] for o in out)
        self.pred = _L(o[7] for o in out)   # True = box SUY ĐOÁN (detector hụt), không phải detection thật


class _Res:
    def __init__(self, img, boxes):
        self.orig_img, self.boxes = img, boxes


def lane_track_stream(model, predict_kw, accept, tracker, dup_iomin=0.75):
    """Chạy detector (model.predict), lọc bằng accept(x1,y1,x2,y2,cls,conf)->bool, bỏ box trùng, rồi gán ID."""
    for i, r in enumerate(model.predict(**predict_kw)):
        dets = []
        if r.boxes is not None and len(r.boxes.xyxy):
            for (x1, y1, x2, y2), c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(),
                                               r.boxes.conf.tolist()):
                if accept(x1, y1, x2, y2, c, cf):
                    dets.append((x1, y1, x2, y2, c, cf))
        if dup_iomin and dup_iomin < 1.0:
            dets = _dedupe(dets, dup_iomin, tracker.stats)
        out = tracker.update(i, dets, r.orig_img)
        yield _Res(r.orig_img, _Boxes(out) if out else None)