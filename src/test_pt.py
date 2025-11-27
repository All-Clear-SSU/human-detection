import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture("./test_video7.mp4")

# 모델 학습된 클래스 수와 이름이 정확히 일치해야 됨
class_names = ["Fire", "Human", "Smoke"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream read failed")
        break

    results = model(frame, conf=0.5)[0]

    vis = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf)

        cls_id = int(box.cls)  # 클래스 ID
        cls_name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"

        # 색상 (클래스별)
        color = (0,255,0) if cls_name == "Human" else \
                (0,0,255) if cls_name == "Fire" else \
                (255,165,0) if cls_name == "Smoke" else (200,200,200)

        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLO PT Stream", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

