import cv2
import numpy as np
import psutil, os, time
from ultralytics import YOLO

class YOLOOnnx:
    def __init__(self, model_path, class_names, input_size=640):
        self.model = YOLO(model_path)  # ultralytics ONNX/PT 모두 가능
        self.class_names = class_names
        self.process = psutil.Process(os.getpid())
        self.input_size = input_size

    def preprocess(self, img):
        # ultralytics 내부에서 처리되므로 간단히 리턴
        return img

    def predict(self, img_path, conf_threshold=0.5, nms_threshold=0.4, save_path="predicted_img.jpg"):
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]

        # Ultralytics 모델 실행
        results = self.model(img, conf=conf_threshold, iou=nms_threshold)[0]

        boxes_list = []
        summary = {name: 0 for name in self.class_names}

        vis = img.copy()

        for box in results.boxes:
            # 좌표
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "Unknown"
            summary[cls_name] += 1

            boxes_list.append({
                "class": cls_name,
                "confidence": float(conf),
                "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            })

            # Draw bounding box
            color = (0,255,0) if cls_name == "Human" else \
                    (0,0,255) if cls_name == "Fire" else \
                    (255,165,0) if cls_name == "Smoke" else (200,200,200)

            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(vis, (x1, label_y - label_height), (x1 + label_width, label_y + 5), color, cv2.FILLED)
            cv2.putText(vis, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Save image
        if save_path:
            cv2.imwrite(save_path, vis)

        result = {
            "image_path": save_path,
            "detections": boxes_list,
            "summary": {**summary, "total_objects": len(boxes_list)}
        }

        return result

    def benchmark(self, img_path, runs=100):
        img = cv2.imread(img_path)

        times, cpu_usages, mem_usages = [], [], []

        for _ in range(runs):
            start_t = time.time()
            _ = self.model(img, conf=0.5)[0]
            end_t = time.time()

            times.append((end_t - start_t) * 1000)
            cpu_usages.append(self.process.cpu_percent(interval=0.0))
            mem_usages.append(self.process.memory_info().rss / (1024*1024))

        avg_time = sum(times) / runs
        avg_cpu = sum(cpu_usages) / runs
        avg_mem = sum(mem_usages) / runs
        fps = 1000 / avg_time

        print(f"\n⏱️ Benchmark Result ({runs} runs)")
        print(f"• Avg Inference Time : {avg_time:.2f} ms")
        print(f"• Estimated FPS      : {fps:.2f} FPS")
        print(f"• Avg CPU Usage      : {avg_cpu:.2f} %")
        print(f"• Avg RAM Usage      : {avg_mem:.2f} MB\n")

