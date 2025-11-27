
import argparse

arg_parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX format")
arg_parser.add_argument("model_path", type=str, help="Path to the YOLO model file (.pt)")

model_path = arg_parser.parse_args().model_path

from ultralytics import YOLO

model = YOLO(model_path)

model.export(
    format="onnx",
    opset=17,
    dynamic=True,        # 동적 배치 지원
    simplify=True        # ONNX 최적화
)
