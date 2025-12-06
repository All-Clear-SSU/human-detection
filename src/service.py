from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import cv2
import numpy as np
import os
from collections import Counter
# yolo_model.py가 같은 디렉토리에 있다고 가정합니다.
from .yolo_model import YOLOOnnx 
# 새로 만든 yolo_pose_model.py에서 YOLOPoseOnnx 클래스를 임포트합니다.
from .yolo_pose_model import YOLOPoseOnnx 

# ----------------- CONFIG -----------------
MODEL_PATH = "./model/best_human.onnx"
COCO_MODEL_PATH = "./model/yolo11n.onnx"
# GENERALIST_MODEL_PATH = "yolo
CLASS_NAMES = ["fire", "human", "smoke"]

# --- POSE MODEL CONFIG ---
POSE_MODEL_PATH = "./model/best_pose.onnx"
POSE_CLASS_NAMES = ["Crawling", "Falling", "Sitting", "Standing"]

SAVE_DIR = "./predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------- INIT -----------------
# 1. 기본 객체 탐지 모델 (fire, human, smoke)
yolo = YOLOOnnx(MODEL_PATH, COCO_MODEL_PATH, CLASS_NAMES)

# 2. 자세 분류 모델 (Standing, Sitting, Fall)
# YOLOPoseOnnx 클래스를 사용하여 초기화합니다.
# 분류 모델의 입력 크기에 맞게 input_shape를 (224, 224) 등으로 변경할 수 있습니다.
pose_model = YOLOPoseOnnx(POSE_MODEL_PATH, POSE_CLASS_NAMES, input_shape=(640, 640)) 

app = FastAPI(title="YOLO ONNX Detection API")

# ----------------- HELPER FUNCTION -----------------
def draw_pose_predictions(img, detections):
    """
    새로운 detections JSON 구조를 기반으로 모든 바운딩 박스와 레이블을 그립니다.
    'human' 클래스에는 'pose' 정보를 추가로 표시합니다.
    """
    img_copy = img.copy()
    for det in detections:
        box = det.get("box")
        if box is None:
            continue
            
        x1, y1, x2, y2 = map(int, [box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)])
        class_name = det.get("class", "Unknown")
        confidence = det.get("confidence", 0)
        
        # 클래스별 색상 설정
        if class_name == "fire":
            color = (0, 0, 255) # Red
        elif class_name == "smoke":
            color = (100, 100, 100) # Gray
        elif class_name == "human":
            color = (0, 255, 0) # Green
        else:
            color = (255, 0, 0) # Blue
            
        # 레이블 생성
        label = f"{class_name} ({confidence*100:.0f}%)"
        
        # 'human' 클래스일 경우, 'pose' 정보 추가
        if class_name == "human":
            pose = det.get('pose', 'Unknown')
            pose_score = det.get('pose_score', 0)
            label = f"Human: {pose} ({pose_score*100:.0f}%)" # 포즈 점수를 표시하도록 수정
            
        # 사각형 및 텍스트 그리기
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        # 텍스트 배경 추가 (가독성)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_copy, (x1, y1 - h - 10), (x1 + w, y1 - 10), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    return img_copy

# ----------------- HEALTH -----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------- STANDARD PREDICT (JSON) -----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    기본 YOLO 모델을 실행하고, 원하시는 JSON 포맷으로 결과를 반환합니다.
    (yolo.predict가 이 포맷을 반환한다고 가정)
    """
    # 임시 파일 저장
    temp_path = os.path.join(SAVE_DIR, f"temp_base_{file.filename}")
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)
    
    result_image_path = os.path.join(SAVE_DIR, f"pred_base_{file.filename}")
    result = yolo.predict(temp_path, conf_threshold=conf_threshold, save_path=result_image_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if "image_path" not in result:
        result["image_path"] = result_image_path

    print(result)
    return JSONResponse(content=result)

# ----------------- STANDARD PREDICT (IMAGE) -----------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    기본 YOLO 모델을 실행하고, 주석이 달린 이미지를 반환합니다.
    """
    temp_path = os.path.join(SAVE_DIR, f"temp_base_img_{file.filename}")
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    result_image_path = os.path.join(SAVE_DIR, f"pred_base_img_{file.filename}")
    result = yolo.predict(temp_path, conf_threshold=conf_threshold, save_path=result_image_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    image_path = result.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Annotated image not found"})
        
    return FileResponse(image_path, media_type="image/jpeg")

# ----------------- POSE PIPELINE (JSON) -----------------
@app.post("/predict_with_pose")
async def predict_with_pose(
    file: UploadFile = File(...), 
    conf_threshold: float = 0.5, 
    pose_conf_threshold: float = 0.3
):
    """
    YOLO (Detection) + YOLOPose (Classification) 파이프라인을 실행하고, 
    'pose'가 추가된 JSON을 반환합니다.
    """
    # 1. 이미지 읽기
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # 2. 임시 원본 파일 저장 (yolo.predict가 파일 경로를 받으므로)
    temp_path = os.path.join(SAVE_DIR, f"temp_pose_{file.filename}")
    cv2.imwrite(temp_path, img)

    # 3. 기본 YOLO 모델 실행 (주석/저장 없이 JSON만 받기)
    yolo_result = yolo.predict(temp_path, conf_threshold=conf_threshold, save_path=None) 
    
    if "detections" not in yolo_result:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": "Invalid base prediction format", "data": yolo_result})

    # 4. 'human' 객체에 대해 Pose 모델 실행
    final_detections = []
    pose_counts = Counter()

    for det in yolo_result.get("detections", []):
        if det.get("class") == "human":
            # 4a. 'human' 박스 좌표로 원본 이미지 크롭
            box = det.get("box", {})
            x1, y1, x2, y2 = map(int, [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")])
            
            # 크롭 (패딩 없이 원본 박스 사용, 필요시 패딩 추가)
            cropped_img = img[y1:y2, x1:x2]
            
            if cropped_img.size == 0:
                det["pose"] = "Unknown (Crop Error)"
                det["pose_score"] = 0.0
                final_detections.append(det)
                continue

            # 4c. Pose 모델 실행 (새로운 YOLOPoseOnnx 클래스 사용)
            # predict_pose는 (class_name, score)를 반환
            detected_pose, pose_score = pose_model.predict_pose(cropped_img, pose_conf_threshold)
            
            # 4e. 'pose' 정보 추가
            det["pose"] = detected_pose
            det["pose_score"] = pose_score # pose 점수도 추가
            pose_counts[detected_pose] += 1
            
        final_detections.append(det)

    # 5. 최종 Summary 업데이트
    base_summary = yolo_result.get("summary", {})
    base_summary["pose_counts"] = dict(pose_counts) # 예: {"Fall": 1, "Standing": 2}

    # 6. 주석이 달린 최종 이미지 생성 및 저장 (JSON에 경로 포함)
    annotated_img = draw_pose_predictions(img, final_detections)
    final_image_path = os.path.join(SAVE_DIR, f"final_pose_{file.filename}")
    cv2.imwrite(final_image_path, annotated_img)
    
    # 7. 임시 원본 파일 삭제
    os.remove(temp_path)

    # 8. 최종 JSON 반환
    return JSONResponse(content={
        "image_path": final_image_path,
        "detections": final_detections,
        "summary": base_summary
    })

# ----------------- POSE PIPELINE (IMAGE) -----------------
@app.post("/predict_image_with_pose")
async def predict_image_with_pose(
    file: UploadFile = File(...), 
    conf_threshold: float = 0.5, 
    pose_conf_threshold: float = 0.3
):
    """
    YOLO + Pose 파이프라인을 실행하고, 'pose' 정보까지 시각화된
    최종 이미지를 반환합니다.
    """
    # 1. 이미지 읽기
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # 2. 임시 원본 파일 저장
    temp_path = os.path.join(SAVE_DIR, f"temp_pose_img_{file.filename}")
    cv2.imwrite(temp_path, img)

    # 3. 기본 YOLO 모델 실행
    yolo_result = yolo.predict(temp_path, conf_threshold=conf_threshold, save_path=None) 
    
    if "detections" not in yolo_result:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": "Invalid base prediction format", "data": yolo_result})

    # 4. 'human' 객체에 대해 Pose 모델 실행
    final_detections = []
    
    for det in yolo_result.get("detections", []):
        if det.get("class") == "human":
            box = det.get("box", {})
            x1, y1, x2, y2 = map(int, [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")])
            
            cropped_img = img[y1:y2, x1:x2]
            
            if cropped_img.size == 0:
                det["pose"] = "Unknown"
                final_detections.append(det)
                continue

            # 4c. Pose 모델 실행 (새로운 클래스 사용)
            detected_pose, pose_score = pose_model.predict_pose(cropped_img, pose_conf_threshold)
            
            det["pose"] = detected_pose
            det["pose_score"] = pose_score # 이미지 레이블에 점수를 쓰기 위해
            
        final_detections.append(det)

    # 5. 주석이 달린 최종 이미지 생성 및 저장
    annotated_img = draw_pose_predictions(img, final_detections)
    final_image_path = os.path.join(SAVE_DIR, f"final_pose_image_{file.filename}")
    cv2.imwrite(final_image_path, annotated_img)
    
    # 6. 임시 원본 파일 삭제
    os.remove(temp_path)

    # 7. 최종 이미지 반환
    return FileResponse(final_image_path, media_type="image/jpeg")

# ----------------- RUN -----------------
if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
