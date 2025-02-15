import numpy as np
import cv2 as cv2
import os
from flaskr.facemakeup.test import evaluate
from flaskr.facemakeup.makeup import makeup


def segmentation(path):
    # path : folder name(id)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 절대경로
    image_path = os.path.join(path, 'face.jpg')
    cp = os.path.join(current_dir, 'cp/79999_iter.pth')  # 상대경로->절대경로로

    # segmentation(parsing)
    img = cv2.imread(image_path)  # 우선 이미지 읽어들이기
    img512 = cv2.resize(img, (512, 512))
    image_path = os.path.join(path, 'face512.jpg')
    cv2.imwrite(image_path, img512)
    parsing = evaluate(image_path, cp)  # test.py의 evaluate() 함수.
    if(12 not in parsing):
        return 'fail'
        
    y, x = img.shape[:2]
    # parsing_img_path = os.path.join(path, 'parsing22.jpg')
    # cv2.imwrite(parsing_img_path, parsing)
    # cv2.imshow("test", parsing)

    # img.shape[] : [Y축, X축, 채널수] 순서 중 Y,X 2개만 가져옴(**원본 img크기로 변환**), 픽셀사이 값 보간.
    parsing = cv2.resize(
        parsing, (x, y), interpolation=cv2.INTER_NEAREST)

    # save parsing image
    parsing_img_path = os.path.join(path, 'parsing.jpg')
    cv2.imwrite(parsing_img_path, parsing)
    return 'success'
