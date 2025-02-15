#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import os
from flaskr.facemakeup.model import BiSeNet
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

import torch.onnx


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return vis_parsing_anno
    # return vis_im

# 학습시키기(학습시켜놓은걸로 평가하기)
def evaluate(image_path, cp):

    # if not os.path.exists(respth):
    #     os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes) # BiSeNet으로 net이리는 인스턴스 생성됨. 인자로 19 넣어서 만듦.
    net.cuda() # Tensor들을 GPU로 보내기 => 변환시 지워야??
    net.load_state_dict(torch.load(cp))
    net.eval()


    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        #segmentation.py에서 cv2.imshow("",image_path) --> 원본 그대로 
        img = Image.open(image_path) 
        #print(np.shape(img))
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        
        img = img.cuda() # 변환시 GPU관련 코드 삭제??
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        # vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=osp.join(respth, dspth))
        return parsing

if __name__ == "__main__":
    #evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img/116.jpg', cp='79999_iter.pth')
    evaluate('./imgs/116.jpg', cp='cp/79999_iter.pth') #이렇게 바꾸면 오류는 해결. line74 주석처리하면 아무것도 출력 X


