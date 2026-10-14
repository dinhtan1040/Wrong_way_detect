# Phát hiện phương tiện đi ngược chiều (dùng VisDrone + video tự quay)

Pipeline: Detection (YOLO11) → Tracking (tracker riêng theo làn `lane_tracker.py`, hoặc BoT-SORT của Ultralytics) → Trajectory (điểm chạm đất = giữa cạnh dưới bbox) → Direction estimation (so với chiều chuẩn của từng làn).

```
video → YOLO (detection) → lọc theo làn → tracker (ID, nhớ và suy đoán vị trí khi xe bị che)
      → quỹ đạo (điểm chạm đất) → hướng (vector đầu→cuối của cửa sổ) → cos với chiều chuẩn của làn
      → ngược liên tục ≥ min_viol_s ? → cảnh báo (events.csv)
```
Không train từ đầu: dùng model public (COCO `yolo11m.pt`), train tiếp trên VisDrone, rồi (tuỳ chọn) fine-tune trên khung hình tự quay.
Hệ thống tính hướng trên toạ độ ảnh (pixel), nên **không cần calibrate camera**; chỉ cần camera cố định.

## 0. Cài đặt và kiểm tra
Cài PyTorch (đúng CUDA) **trước**, rồi mới cài các thư viện còn lại. Nếu làm ngược lại, `ultralytics` có thể kéo bản PyTorch chỉ-CPU về và lệnh cài CUDA sau đó không thay thế nó.
```
python -m pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python test_logic.py        # phải in "TEST OK" và "TEST CLASSES/EVAL OK"
```
GPU đã dùng: RTX 3050 Ti 4 GB (torch 2.5.1+cu121, Ultralytics 8.4.173). Hết VRAM thì giảm `batch` hoặc dùng `yolo11s.pt`.

## 1. Cấu trúc repo, dữ liệu và model
Repo chỉ chứa **code, cấu hình, bảng kết quả nhỏ và tài liệu**. Video và ảnh tự quay chứa người và biển số xe, lại rất nặng, nên **không nằm trong Git**.
```
*.py                    mã nguồn (xem "Danh sách file" ở cuối)
configs/                cấu hình làn theo từng video (JSON)
configs_example/        cấu hình mẫu
gt_example.csv          mẫu đáp án sự kiện
results/                bảng kết quả, results.csv và args.yaml của các lần train
docs/figures/           hình dùng trong báo cáo (đã làm mờ biển số, khuôn mặt)
data/  runs/  dataset/  KHÔNG đưa lên Git (xem .gitignore)
```
- Dữ liệu tự quay: `<LIEN_KET_DRIVE>` (đặt vào `data/videos/`).
- Trọng số đã train (`train-7`, `ft`): `<LIEN_KET_RELEASES>` (đặt vào `models/`).

## 2. Ba mức dữ liệu và ba model
| Mức | Dữ liệu | Đo gì | Lệnh |
|---|---|---|---|
| A | VisDrone (val / test-dev) | detector (AP50 theo nhóm lớp) | `05_eval_detector.py` |
| B | Video tự quay, chưa fine-tune | detector trên khung tự quay + sự kiện ngược chiều | `05_eval_detector.py`, `run_models.py` |
| C | Sau khi thêm dữ liệu tự quay / fine-tune | như B | như B |

Model: **M0** = COCO `yolo11m.pt`; **M1** = train trên VisDrone (`runs/detect/train-7/weights/best.pt`); **M2** = M1 fine-tune trên khung tự quay.
VisDrone là ảnh drone, không có nhãn ngược chiều, nên chỉ dùng cho mức detector (A) và làm detector khởi tạo tốt hơn. Đánh giá sự kiện chỉ làm trên video của nhóm.
Tuỳ chọn thêm một mốc đối chứng: fine-tune cùng dữ liệu và cùng siêu tham số nhưng xuất phát từ COCO (`yolo11m.pt`) thay vì `train-7`, để thấy điểm xuất phát có ảnh hưởng không.

## 3. Train trên VisDrone (đã làm)
```
yolo train model=yolo11m.pt data=VisDrone.yaml epochs=20 imgsz=640 batch=4 workers=2 device=0
```
Kết quả val (548 ảnh, imgsz=640): car mAP50 0.808, motor mAP50 0.486 (recall 0.48). Xe máy là lớp yếu nhất vì nhỏ.
**Báo cáo mức A bằng split test-dev** (kiểm tra `VisDrone.yaml` có mục `test`), không dùng val vì val đã dùng để chọn checkpoint.

**ID lớp khác nhau giữa các model** (COCO motorcycle=3, VisDrone motor=9). Mọi script chọn và gộp lớp theo TÊN (`classes.py`):
motor/motorcycle → motorcycle; car/van → car; bus; truck; bicycle. `tricycle`, `awning-tricycle` và lớp người không được đánh giá.
Cần kiểm tra bằng mắt rằng nhãn `motor` của VisDrone bao cả người lái; cách gán khác sẽ làm số liệu M0 và M1 không hoàn toàn so sánh được.

## 4. Đánh giá detector (mức A và B/C)
```
# Mức A
python 05_eval_detector.py --data VisDrone.yaml --split test --imgsz 640 \
    --weights yolo11m.pt runs/detect/train-7/weights/best.pt --names M0_coco M1_visdrone --out eval_visdrone_test.csv

# Khung tự quay (tập test chia theo video, tạo bằng 04_make_dataset.py)
python 05_eval_detector.py --data dataset/data.yaml --split test --imgsz 960 \
    --weights yolo11m.pt runs/detect/train-7/weights/best.pt runs/train/ft/weights/best.pt \
    --names M0_coco M1_visdrone M2_finetuned --out eval_own_test.csv
```
In AP50 theo nhóm, và với xe máy: Precision/Recall/F1 + Recall theo cỡ (cao <32, 32–96, ≥96 px) để phân tích khoảng cách. Dùng imgsz bằng lúc train (640) cho VisDrone.
Số liệu có thể lệch nhẹ so với `yolo val` do gộp lớp. `detector_eval.py` mới chỉ được kiểm tra bằng dữ liệu giả; hãy đối chiếu một lần với `yolo val` trên VisDrone.
Đường dẫn `runs/train/ft/...` khớp với lệnh fine-tune ở mục 8 (`project=runs/train name=ft`); nếu bạn train với đường dẫn khác thì sửa cho khớp.

## 5. Video, làn và chiều đúng
Đặt tên `<điều_kiện>_<số>.mp4` (phần trước `_` đầu tiên dùng để nhóm kết quả; tên như `Wrong_way_1` sẽ bị nhóm thành `Wrong`, nên tránh nếu cần nhóm theo điều kiện). Camera cố định, không zoom số.
```
data/videos/*.mp4   configs/<tên_video>.json (hoặc configs/default.json)   gt.csv   runs/
```
```
python 02_draw_roi.py --video data/videos/Rain_1.mp4 --out configs/Rain_1.json
python preview_config.py --video data/videos/Wrong_way_1.mp4 --config configs     # kiểm tra polygon + mũi tên chiều ĐƯỢC PHÉP
```
`configs_example/Wrong_way_1.json` là cấu hình thực tế của nhóm (video dọc 1080x1920, 2 làn). Chiều đúng cần xác nhận ngoài thực tế (biển cấm đi ngược / mũi tên keep-right).
Cấu hình JSON có thể thêm `ignore_zones` (danh sách polygon, cùng hệ toạ độ với làn) để bỏ qua vật tĩnh hay bị nhận nhầm như cột biển báo.
Cắt đoạn ngắn để thử: `python cut_clip.py --video data/videos/Wrong_way_1.mp4 --start 420 --end 450` (ghi vào `data/clips/<cùng tên>.mp4`).

## 6. Chạy hệ thống
```
python detect_wrong_way.py --video data/videos --config configs \
    --weights runs/detect/train-7/weights/best.pt --out runs/M1_visdrone --imgsz 960
```
Mỗi video cho ra trong thư mục `--out`: `<tên>_events.csv` (sự kiện ngược chiều), `_tracks.csv` (mọi track), `_meta.json` (tham số, tốc độ, thống kê bộ lọc và tracker), `_out.mp4` (video vẽ khung, tắt bằng `--no-video`), `_debug.csv` (với `--debug`).

**Hai chế độ tracking** (`--tracking`):
- `lane` (mặc định): tracker riêng trong `lane_tracker.py`. Chỉ cấp ID cho xe trong vùng làn; xác nhận track sau vài khung nên box ma không sinh ID; chống trùng vật thể; xe đứng yên được nhớ lâu hơn; khi detector hụt hoặc xe bị che thì **suy đoán vị trí theo hướng chạy** (box vàng cam, nhãn có dấu `~`) và nhận lại đúng ID khi xe xuất hiện lại. Khung suy đoán **không** làm tăng bộ đếm vi phạm nên không thể tự tạo cảnh báo. Tham số: `--lost-s`, `--coast-s`, `--coast-static-s`, `--new-conf`, `--gate-frac`, `--dup-iomin`, `--no-appearance`.
- `ultra`: BoT-SORT / ByteTrack của Ultralytics (`--tracker track_static.yaml`), kèm bộ nối ID đơn giản (`--link-gap-s`, `--link-radius-frac`). Dùng để đối chứng.

**Tham số hướng** (theo giây nên không phụ thuộc fps; giá trị mặc định trong code): `--window-s 0.5 --min-disp-frac 0.3 --cos-thresh -0.5 --min-viol-s 0.3`. **cos_thresh càng âm càng khó báo.** Hệ thống báo khi xe ngược chiều **liên tục** ít nhất `min_viol_s` giây (tính theo thời gian, không theo số khung).
Bộ lọc box ma: `--max-box-frac` (bỏ box quá to), `--drop-classes bus`, `--lane-margin-frac` (lề quanh polygon).
`--conf 0.15` là ngưỡng đưa vào tracker; `--new-conf` (mặc định 0.4) là ngưỡng để được cấp ID mới.
`--include-person` chỉ dành cho model COCO (xe máy hay bị gán `person` nhìn từ trên cao). Model VisDrone hoặc đã fine-tune không cần.
`--no-show` tắt cửa sổ xem trực tiếp (cần khi chạy trên server hoặc Colab); `--device 0` chọn GPU.
Xe nhỏ ở xa: thử `--imgsz 1280`. Model train ở 640 có thể hoạt động tốt nhất quanh 640–960, hãy quét (mục 9).

### Không bắt được xe ngược chiều? Chẩn đoán
```
python detect_wrong_way.py --video data/clips/Wrong_way_1.mp4 --config configs --out runs/debug --stride 1 --debug
python diagnose.py --debug-csv runs/debug/Wrong_way_1_debug.csv --t-from 420 --t-to 450
```
Phễu: track → trong polygon → đủ dài → dịch chuyển đủ lớn → cos < ngưỡng → ngược liên tục. Bước nào mất nhiều nhất là nguyên nhân.
Cột `pred` trong `_debug.csv` đánh dấu khung suy đoán. Ở chế độ `ultra`, số ID có thể tăng nhanh vì tracker cấp ID cả cho box về sau bị lọc; số track ngắn (dưới ~0,5 s) trong `_tracks.csv` mới là dấu hiệu tracker bị vỡ thật. Ở chế độ `lane`, ID chỉ tăng khi track được xác nhận.

## 7. Đánh giá sự kiện (FP/FN)
`gt.csv` (mẫu `gt_example.csv`): mỗi dòng một xe ngược chiều `video,t_start,t_end` (giây). Video có nhiều sự kiện thì nhiều dòng; video không có thì không cần dòng.
```
python evaluate.py --pred runs/M1_visdrone --gt gt.csv --tol 2.0
```
So sánh nhiều model một lệnh:
```
python run_models.py --models M0_coco=yolo11m.pt M1_visdrone=runs/detect/train-7/weights/best.pt \
    --video-dir data/videos --config configs --gt gt.csv --out runs/models --imgsz 960 --include-person-for M0_coco
```
Ra `runs/models/models_results.csv`: TP/FP/FN, precision, recall, F1, FP/giờ, độ trễ trung bình. Một xe bị đổi ID hoặc xe 2 người có thể sinh thêm cảnh báo (FP): nêu rõ trong báo cáo.

## 8. Fine-tune (mức C)
```
python 01_extract_frames.py --video-dir data/videos --out data/frames --every-sec 1.5
python 03_autolabel.py --frames data/frames --weights runs/detect/train-7/weights/best.pt
# SỬA TAY nhãn nháp (labelImg / CVAT / Roboflow / label_review.py, định dạng YOLO; classes.txt đã tạo)
python 04_make_dataset.py --frames data/frames --out dataset --weights runs/detect/train-7/weights/best.pt --test-videos Rain_3 --val-videos Rain_2
yolo detect train model=runs/detect/train-7/weights/best.pt data=dataset/data.yaml imgsz=960 epochs=30 batch=4 patience=10 freeze=10 device=0 project=runs/train name=ft
```
Ghi đúng tham số bạn thực sự đã dùng (xem `args.yaml` của lần train, ví dụ `imgsz`, `epochs`, `freeze`) để người khác tái lập được.

Quy ước nhãn: **một khung duy nhất cho xe máy + người lái**, nhãn `motor`. Chia train/val/test **theo VIDEO**.
**Chống rò rỉ dữ liệu:**
- Không dùng khung của đoạn có sự kiện ngược chiều để train.
- Clip trích từ một video khác (ví dụ `Wrong_way` là đoạn của `Wrong_way_2`) không được vừa nằm trong train vừa được dùng để đánh giá.
- Nhãn nháp do model sinh ra không dùng cho tập test; nhãn test phải do người gán.
Lưu ý 04_make_dataset.py lấy tên lớp từ `--weights`, phải trùng với 03_autolabel.py.
Khi đánh giá, nhớ rằng val dùng để chọn checkpoint nên số trên val lạc quan hơn số trên test.

## 9. Phân tích yếu tố
```
python run_sweep.py --video-dir data/videos --config configs --weights runs/detect/train-7/weights/best.pt \
    --gt gt.csv --out runs/sweep --imgsz 640 960 1280 --strides 1 2 3 6
python make_degraded.py --video-dir data/videos --out data/videos_noise20 --mode noise --level 20
python detect_wrong_way.py --video data/videos_noise20 --config configs --weights runs/detect/train-7/weights/best.pt --out runs/noise20 --no-video
python evaluate.py --pred runs/noise20 --gt gt.csv
```
Độ phân giải, fps (stride), nhiễu, mờ, tối, nén: ở trên. Ánh sáng: nhóm theo tên video (hiện toàn bộ là mưa ban ngày nên chưa so sánh được ánh sáng). Khoảng cách/tốc độ: chia theo cỡ bbox hoặc vị trí y và độ dịch chuyển trong `*_tracks.csv`. Góc camera: cần quay thêm một góc khác.
Mỗi lần chỉ đổi một yếu tố, dùng `--out` riêng, và chọn tham số trên val rồi báo cáo trên test.

## 10. Kết quả chính (điền sau khi chạy, mỗi số phải truy được về một file)
| Mức | Model | Dữ liệu | AP50 car / motor | TP / FP / FN (sự kiện) | Precision / Recall |
|---|---|---|---|---|---|
| A | M0 | VisDrone test-dev | … | — | — |
| A | M1 | VisDrone test-dev | … | — | — |
| B | M0 | video tự quay | … | … | … |
| B | M1 | video tự quay | … | … | … |
| C | M2 | video tự quay (test) | … | … | … |

## 11. Lưu ý khi báo cáo
- Dữ liệu chỉ có một vị trí, một thời tiết (mưa), vài sự kiện ngược chiều, nên số F1/Recall có khoảng tin cậy rất rộng; hãy báo số TP/FP/FN tuyệt đối, đừng chỉ báo phần trăm.
- Nêu định nghĩa "phát hiện đúng" (`--tol`), chiều đúng của từng làn, và phép gộp lớp (van→car, tricycle bỏ qua).
- Nêu rõ phần tái sử dụng (YOLO/Ultralytics, VisDrone) và phần nhóm tự làm (lọc làn, tracker riêng, logic hướng, đánh giá).
- Kết luận chỉ đúng cho vị trí camera đã quay.
- Quyền riêng tư và an toàn: không đưa video chứa người hay biển số lên repo công khai; không dàn dựng xe chạy ngược thật khi quay.

## 12. Thành viên
| Họ tên | Phần việc |
|---|---|
| `<tên>` | `<ví dụ: quay và gán nhãn dữ liệu>` |
| `<tên>` | `<ví dụ: detector và fine-tune>` |
| `<tên>` | `<ví dụ: tracking và logic hướng>` |

## 13. Nguồn và giấy phép
- Ultralytics YOLO11: giấy phép AGPL-3.0. VisDrone: theo điều khoản của bộ dữ liệu. Ghi trích dẫn và liên kết tới nguồn gốc.
- Mã của nhóm: `<giấy phép>`.

## Danh sách file
`classes.py` (chọn/gộp lớp theo tên) · `detect_wrong_way.py` (pipeline) · **`lane_tracker.py` (tracker riêng, bắt buộc: `detect_wrong_way.py` import file này)** · `evaluate.py` (sự kiện) · `detector_eval.py` + `05_eval_detector.py` (detector) · `run_models.py` · `run_sweep.py` · `make_degraded.py` · `diagnose.py` · `preview_config.py` · `cut_clip.py` · `01_extract_frames.py` · `02_draw_roi.py` · `03_autolabel.py` · `04_make_dataset.py` · `test_logic.py`
Công cụ gán nhãn (tuỳ chọn, nếu dùng): `label_review.py` (sửa nhãn, có dự đoán sẵn bằng model) · `prelabel_visdrone.py` · `split_by_clip.py` · `sample_frames.py` · `mix_visdrone.py`