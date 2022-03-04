import cv2
import torch
import os

model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=True)
model.eval()

img = cv2.imread("data/images/bus.jpg")

results = model(img)

results.print()
results.show()
