import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input shape: {input_shape}")

# 문자열이나 None을 1로 변환
dummy_shape = [1 if (dim is None or isinstance(dim, str)) else dim for dim in input_shape]

# 더미 입력 생성
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 추론
outputs = session.run(None, {input_name: dummy_input})

# 출력 정보 확인
print(f"Number of outputs: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"Output[{i}] type: {type(out)}, shape: {out.shape}")
    print(f"Sample values: {out.flatten()[:10]}")

