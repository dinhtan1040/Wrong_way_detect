"""Chọn lớp theo TÊN, không theo ID.

Model COCO (yolo11m.pt) và model train trên VisDrone có bộ lớp KHÁC NHAU:
    COCO    : 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck, 0 person
    VisDrone: 0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle,
              7 awning-tricycle, 8 bus, 9 motor
Lọc theo ID cố định sẽ chọn nhầm lớp khi đổi model, nên mọi script dùng các hàm dưới đây.
"""

VEHICLE_NAMES = {"bicycle", "car", "van", "truck", "bus", "motorcycle", "motor",
                 "tricycle", "awning-tricycle"}
PERSON_NAMES = {"person", "pedestrian", "people"}

# Nhóm chung để so sánh giữa các bộ lớp khác nhau (dùng trong 05_eval_detector.py).
# tricycle / awning-tricycle không có trong COCO nên không đưa vào so sánh.
GROUP_OF = {
    "bicycle": "bicycle",
    "car": "car", "van": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle", "motor": "motorcycle",
}
GROUPS = ["motorcycle", "car", "bus", "truck", "bicycle"]


def _items(names):
    return names.items() if isinstance(names, dict) else enumerate(names)


def resolve_ids(names, wanted):
    """ID các lớp có tên (chữ thường) nằm trong `wanted`."""
    return sorted(int(i) for i, n in _items(names) if str(n).lower() in wanted)


def vehicle_ids(names, include_person=False):
    """ID lớp phương tiện của model (thêm lớp người nếu include_person)."""
    wanted = set(VEHICLE_NAMES) | (PERSON_NAMES if include_person else set())
    ids = resolve_ids(names, wanted)
    if not ids:
        raise SystemExit("Model không có lớp phương tiện nào (tên lớp: %s)" % list(dict(_items(names)).values())[:12])
    return ids


def group_of(name):
    """Nhóm chung của một tên lớp, hoặc None nếu không đánh giá."""
    return GROUP_OF.get(str(name).lower())
