import torch
from utils.datasets import LoadImages

model = torch.hub.load('.', 'custom', path='weights/yolov5x-e8-ds2.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5x-e8-ds2.pt')

print(model.names)