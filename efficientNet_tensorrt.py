import torch
from PIL import Image
from torchvision.transforms import transforms

from torch2trt import torch2trt, TRTModule
import torchvision.models as models

# EfficientNet 모델 로드
model = models.efficientnet_b0(pretrained=True)
model.eval()

# 입력 데이터 전처리
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 입력 데이터 생성
image_path = "image.jpg"  # 예시 이미지 경로
image = Image.open(image_path)
input_data = preprocess(image).unsqueeze(0)

# PyTorch 모델을 TensorRT 엔진으로 변환
trt_model = torch2trt.torch2trt(model, [input_data])

# TensorRT 엔진 저장
trt_path = "efficientnet.trt"
torch.save(trt_model.state_dict(), trt_path)

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# TensorRT 엔진 로
model = TRTModule()
trt_path = "efficientnet.trt"
model.load_state_dict(torch.load(trt_path))

# 추론 실행
trt_outputs = model(x)

# 추론 결과 출력
output = trt_outputs[0]
_, predicted_idx = torch.max(output, 0)
print("Predicted class index:", predicted_idx.item())
