import torch
import torchvision.models as models
import torchvision.transforms as transforms
import tensorrt as trt
from PIL import Image
import torch2trt
from torch2trt import TRTModule
import time
import numpy as np

# 추론을 위한 입력 데이터 생성
input_data = torch.ones((1, 3, 224, 224)).cuda()

# 모델 로드
trt_path = "../../lab-equip-detection-kiosk/yolov7/models/efficientnet.trt"

# model = TRTModule()
# model.load_state_dict(torch.load(trt_path))
model = torch.load("EfficientNet_05.15.15.06_224.pt")
model.eval()
model.cuda()

# 추론 실행
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

for _ in range(10):
    _ = model(input_data)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(input_data)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

# 추론 시간 계산
inference_time = np.average(timings)
print("Inference Time:", inference_time, "seconds")
