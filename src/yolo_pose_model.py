import cv2
import numpy as np
import onnxruntime as ort

class YOLOPoseOnnx:
    """
    ONNX 기반의 자세 '분류(Classification)' 모델을 처리하기 위한 전용 클래스.
    크롭된 'human' 이미지를 입력받아 'Standing', 'Sitting', 'Fall' 중 하나로 분류합니다.
    """

    def __init__(self, model_path, class_names, input_shape=(640, 640)):
        """
        모델과 세션을 초기화합니다.
        :param model_path: .onnx 모델 파일 경로
        :param class_names: ["Crawling", "Falling", "Sitting", "Standing"]

        :param input_shape: 모델이 요구하는 입력 이미지 크기 (정사각형 가정)
        """
        self.class_names = class_names
        self.input_shape = input_shape

        # ONNX 런타임 세션 생성
        self.session = ort.InferenceSession(model_path, 
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # 모델의 입력/출력 이름 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Pose Model loaded: {model_path}")
        print(f"Pose Input: {self.input_name}, Output: {self.output_name}")
        print(f"Pose Classes: {self.class_names}")


    def preprocess_image(self, image_bgr):
        """
        입력 이미지를 모델에 맞게 전처리합니다. (YOLOv5/v8 기준)
        """
        # 1. 레터박스로 리사이즈
        h, w = image_bgr.shape[:2]
        target_h, target_w = self.input_shape
        
        r = min(target_h / h, target_w / w)
        new_h, new_w = int(h * r), int(w * r)
        
        resized_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. 패딩 추가
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 3. BGR -> RGB
        image_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # 4. HWC -> CHW
        image_chw = np.transpose(image_rgb, (2, 0, 1))
        
        # 5. 정규화 (0-255 -> 0.0-1.0)
        image_normalized = image_chw.astype(np.float32) / 255.0
        
        # 6. 배치 차원 추가 (CHW -> NCHW)
        input_tensor = np.expand_dims(image_normalized, axis=0)
        
        return input_tensor

    def predict_pose(self, cv2_image, conf_threshold=0.3):
        """
        크롭된 CV2 이미지를 입력받아, 가장 확률이 높은 자세와 점수를 반환합니다.
        
        :param cv2_image: 크롭된 'human' 이미지 (numpy array)
        :param conf_threshold: 최소 신뢰도
        :return: (str: "Fall", float: 0.95)
        """
        if cv2_image is None or cv2_image.size == 0:
            return "Unknown (Image Error)", 0.0

        # 1. 이미지 전처리
        input_tensor = self.preprocess_image(cv2_image)

        # 2. ONNX 추론 실행
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 3. 출력 처리 (분류 모델 가정)
        # 출력이 [[score1, score2, score3]] 형태라고 가정
        scores = outputs[0][0] 
        print(f"Pose model scores: {scores}")
        
        # 4. 가장 높은 점수의 인덱스 찾기 (ArgMax)
        top_class_id = np.argmax(scores)
        top_score = float(scores[top_class_id])

        if top_score < conf_threshold:
            return "Unknown (Low Conf)", top_score
        
        top_class_name = self.class_names[top_class_id]

        return top_class_name, top_score
