import os
import cv2
from ultralytics import YOLO
import supervision as sv

# --- 경로 설정 ---
VIDEO_DIR = "./videos"
VIDEO_FILE = "fire_human_network_server_h264.mp4"
SAVE_DIR = "./predictions/yolo11_" + VIDEO_FILE.split('.')[0]
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 모델 로드 ---
model = YOLO("yolo11x.pt")

# --- person class_id 찾기 ---
person_id = [k for k, v in model.names.items() if v == "person"][0]
print(f"Detected person class_id = {person_id}")

# --- 비디오 열기 ---
video_path = os.path.join(VIDEO_DIR, VIDEO_FILE)
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_path = os.path.join(SAVE_DIR, VIDEO_FILE)
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {VIDEO_FILE} ({frame_count} frames)...")

# --- Annotators ---
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)
    boxes = results[0].boxes  # YOLO box structure

    # Supervision Detections 변환
    detections = sv.Detections(
        xyxy=boxes.xyxy.cpu().numpy(),
        confidence=boxes.conf.cpu().numpy(),
        class_id=boxes.cls.cpu().numpy().astype(int)
    )

    # --- 🔥 person만 남기기 ---
    mask = detections.class_id == person_id
    detections = detections[mask]

    # --- 라벨 생성 ---
    COCO_CLASSES = model.names
    labels = [
        f"{COCO_CLASSES[c]} {conf:.2f}"
        for c, conf in zip(detections.class_id, detections.confidence)
    ]

    # 빈 프레임이면 바로 저장
    annotated_frame = frame.copy()
    if len(detections) > 0:
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    # 저장
    out.write(annotated_frame)

cap.release()
out.release()

print(f"Saved annotated video to: {out_path}")

