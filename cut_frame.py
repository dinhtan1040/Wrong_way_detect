import sys, cv2

video = sys.argv[1]
sec = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
out = sys.argv[3] if len(sys.argv) > 3 else "frame.jpg"

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
ok, f = cap.read()
cap.release()
if not ok:
    raise SystemExit("Không đọc được frame, thử giây nhỏ hơn")
cv2.imwrite(out, f)
print("Đã lưu", out, f.shape)