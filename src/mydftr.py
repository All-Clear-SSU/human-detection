import os
import cv2
from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image

# --- 경로 설정 ---
VIDEO_DIR = "./videos"
VIDEO_FILE = "fire_human_network_server_h264.mp4"
SAVE_DIR = "./predictions/rf_" + VIDEO_FILE.split('.')[0]
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 모델 로드 ---
model = RFDETRBase()
# optimize_for_inference() 제거, 바로 predict 사용

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # RF-DETR inference
    detections = model.predict(pil_img, threshold=0.5)

    # Labels
    from rfdetr.util.coco_classes import COCO_CLASSES
    labels = [
        f"{COCO_CLASSES[class_id]} {conf:.2f}"
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]

    # Annotation
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    # 저장
    out.write(annotated_frame)

cap.release()
out.release()
print(f"Saved annotated video to {out_path}")

