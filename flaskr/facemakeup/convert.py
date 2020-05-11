import torch
import torchvision
from flaskr.facemakeup.test import evaluate
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))  # 절대경로
cp = os.path.join(current_dir, 'cp/79999_iter.pth')  # 상대경로->절대경로로

# segmentation(parsing)
image_path = os.path.join(current_dir, 'iphone_back.jpg')
img = cv2.imread(image_path)  # 우선 이미지 읽어들이기

dummy_input = img
model = evaluate()  # test.py의 evaluate() 함수.

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)