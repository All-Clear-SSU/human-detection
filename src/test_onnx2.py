import cv2
from ultralytics import YOLO
import os

model = YOLO("best.onnx")

# 테스트할 영상 리스트
videos = [
    "test_video.mp4",
    "test_video2.mp4",
    "test_video3.mp4",
    "test_video4.mp4",
    "test_video5.mp4",
    "test_video6.mov",
    "test_video7.mp4",
    "test_video8.mov",
    "test_video9.mp4"
]

# 모델 클래스
class_names = ["Fire", "Human", "Smoke"]

# 결과 저장 폴더
os.makedirs("results", exist_ok=True)

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue

    # 영상 저장용 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join("results", os.path.basename(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Processing {video_path} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)[0]

        vis = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"

            color = (0,255,0) if cls_name == "Human" else \
                    (0,0,255) if cls_name == "Fire" else \
                    (255,165,0) if cls_name == "Smoke" else (200,200,200)

            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 영상 저장
        out.write(vis)

        # 화면에 보기 (선택)
        # cv2.imshow("YOLO ONNX Stream", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    print(f"Saved result to {out_path}")

cv2.destroyAllWindows()

