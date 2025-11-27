# Guide

## Install Yolo

```sh
pip install -U ultralytics
```

## Human Predict

| Category         | Precision | Finding |
|------------------|-----------|---------|
| Human Prediction | 87%       | 87%     |

```sh
yolo predict model=./80-precision_human_detection.pt source='./human_cctv.png' project='./' imgsz=640 name="predict"
```

## 90-fire_60-human_70-smoke.pt


| Category         | Precision | Finding |
|------------------|-----------|---------|
| Fire Prediction  | 96%       | 97%     |
| Human Prediction | 69%       | 60%     |
| Smoke Prediction | 93%       | 80%     |


```sh
yolo predict model=./90-fire_60-human_70-smoke.pt source='./fire_test.jpg' project='./' imgsz=640 name="predict"
```

## 90-fire_80-human_70-smoke.pt


| Category         | Precision | Finding |
|------------------|-----------|---------|
| Fire Prediction  | 96%       | 97%     |
| Human Prediction | 91%       | 85%     |
| Smoke Prediction | 91%       | 79%     |


```sh
yolo predict model=./90-fire_80-human_70-smoke.pt source='./fire_test.jpg' project='./' imgsz=640 name="predict"
```
