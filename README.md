# Human Detection

## Start Service

Run

```sh
uvicorn src.service:app --host 0.0.0.0 --port 8383
```


백그라운드에서 실행하려면

```sh
bash run_model_test_server.sh
# OR
nohup uvicorn src.service:app --host 0.0.0.0 --port 8383 > server.log 2>&1 &
```

## File Structure

⏱️ **Benchmark Result (1000 runs)**

| Metric             | Value     |
| ------------------ | --------- |
| Avg Inference Time | 26.61 ms  | 
| Estimated FPS      | 37.58 FPS |
| Avg CPU Usage      | 99.895 %  |
| Avg RAM Usage      | 154.31 MB |


## Model Detail

실제 사용되는 모델은  

- "./model/best_human.onnx", 
- "./model/best_pose.onnx"

## Human, Fire, Smoke Detection Test API

## predict

```sh

curl -X POST "http://<server-ip>:<port>/predict?conf_threshold=0.5" \
  -F "file=@human_cctv.png"
```
Returns

```json
{
  "image_path": "predicted_img.jpg",
  "detections": [
    {
      "class": "fire",
      "confidence": 0.92,
      "box": { "x1": 250, "y1": 200, "x2": 400, "y2": 350 }
    },
    {
      "class": "human",
      "confidence": 0.87,
      "box": { "x1": 100, "y1": 150, "x2": 200, "y2": 300 }
    },
    {
      "class": "smoke",
      "confidence": 0.78,
      "box": { "x1": 50, "y1": 220, "x2": 180, "y2": 350 }
    }
  ],
  "summary": {
    "fire_count": 1,
    "human_count": 1,
    "smoke_count": 1,
    "total_objects": 3
  }
}
```

## predic_image

```sh
curl -X POST "http://<server-ip>:<port>/predict_image" \
  -F "file=@human_cctv.png" \
  --output output.jpg
```


Returns

```sh
image
```

## Human Pose Estimation Pipeline API

## predict_with_pose

(기본 탐지 + 자세 추정 파이프라인 / JSON 반환)

```sh
curl -X POST "http://<server-ip>:<port>/predict_with_pose?conf_threshold=0.5&pose_conf_threshold=0.3" \
  -F "file=@human_cctv.png"
```


Returns
```sh

{
  "image_path": "predictions/final_pose_human_cctv.png",
  "detections": [
    {
      "class": "fire",
      "confidence": 0.92,
      "box": { "x1": 250, "y1": 200, "x2": 400, "y2": 350 }
    },
    {
      "class": "human",
      "confidence": 0.87,
      "box": { "x1": 100, "y1": 150, "x2": 200, "y2": 300 },
      "pose": "Fall", 
      "pose_score": 0.95
    },
    {
      "class": "human",
      "confidence": 0.75,
      "box": { "x1": 500, "y1": 180, "x2": 600, "y2": 400 },
      "pose": "Standing",
      "pose_score": 0.88
    },
    {
      "class": "smoke",
      "confidence": 0.78,
      "box": { "x1": 50, "y1": 220, "x2": 180, "y2": 350 }
    }
  ],
  "summary": {
    "fire_count": 1,
    "human_count": 2,
    "smoke_count": 1,
    "total_objects": 4,
    "pose_counts": {
      "Fall": 1,
      "Standing": 1
    }
  }
}
```


## predict_image_with_pose

(기본 탐지 + 자세 추정 파이프라인 / 이미지 반환)

```sh
curl -X POST "http://<server-ip>:<port>/predict_image_with_pose" \
  -F "file=@human_cctv.png" \
  --output output_with_pose.jpg
```


Returns

```sh
image
```


(모든 객체와 Human의 Pose가 함께 시각화된 이미지를 반환합니다.)

## FastAPI -> WebUI

[webui](http://49.142.15.26:5557/docs#/default/predict_image_predict_image_post)


![](/home/light/Pictures/resources/20251101_225003.png)


Click Try it out -> Choose File -> Execute

![](/home/light/Pictures/resources/20251101_225040.png)


![](/home/light/Pictures/resources/20251101_225242.png)




## Predict Image With Pose Class

"Crawling", "Falling", "Sitting", "Standing"

Pose model scores: [    0.30319     0.56544    0.060098    0.071273]

![](./output_with_pose.jpg)


## Predictions

예측한 결과가"./predictions" 폴더에 저장됩니다.

## Comparison With Other Models

재난 특화


## Other Models with My Model Comparison


| Model       | Class | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|-------------|--------|---------|-----------|--------|--------|--------|-----------|
| YOLO11x     | human  | 2160    | 6144      | 0.569 | 0.475 | 0.425 | 0.208     |
| YOLO11n     | human  | 2160    | 6144      | 0.479 | 0.369 | 0.309 | 0.132     |
| YOLO11s     | human  | 2160    | 6144      | 0.530 | 0.403 | 0.349 | 0.158     |
| **Our Model** | **human** | **2160** | **6144** | **0.925** | **0.876** | **0.937** | **0.641** |


## Hybrid Model Benchmark

- yolo11x
```sh
📊 Benchmark Result
• Avg Processing Time : 597.26 ms
• Estimated FPS       : 1.67 FPS
```

- yolo11n
```sh
📊 Benchmark Result
• Avg Processing Time : 108.66 ms
• Estimated FPS       : 9.20 FPS
```
