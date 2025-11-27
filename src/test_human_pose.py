import cv2
import os
from src.service import yolo, pose_model, draw_pose_predictions

VIDEO_DIR = "./videos"       # 영상이 있는 폴더
SAVE_DIR = "./predictions"   # 결과 영상 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

def process_video(video_file: str):
    video_path = os.path.join(VIDEO_DIR, video_file)
    print("the path exists: " + os.path.exists(video_path).__str__())
    cap = cv2.VideoCapture(video_path)
    print("the video is opened: " + cap.isOpened().__str__())

    # 영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(SAVE_DIR, f"pose_{video_file}")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Processing {video_file}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # cv2.imshow("Frame", frame)
        # YOLO 탐지
        temp_img_path = os.path.join(SAVE_DIR, "temp_frame.jpg")
        cv2.imwrite(temp_img_path, frame)
        yolo_result = yolo.predict(temp_img_path, conf_threshold=0.5, save_path=None)

        # Pose 적용
        final_detections = []
        for det in yolo_result.get("detections", []):
            if det["class"] == "human":
                box = det["box"]
                x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    pose, score = pose_model.predict_pose(cropped, conf_threshold=0.3)
                    det["pose"] = pose
                    det["pose_score"] = score
            final_detections.append(det)

        # 프레임 주석
        annotated_frame = draw_pose_predictions(frame, final_detections)
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Saved annotated video to {out_path}")

def process_all_videos():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov'))]
    for video_file in video_files:
        process_video(video_file)

# 실행
# process_all_videos()
process_video("fire_human_network_server_h264.mp4")
print("All videos processed.")

