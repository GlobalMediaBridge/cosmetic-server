import os
import cv2
import numpy as np
from skimage.filters import gaussian
import os


def lip(image, parsing, part=12, color=[230, 50, 20]):
    b, g, r = color      # ex) [230, 50, 20] == [b, g, r]
    # np.zeros_like() : 다른 배열과 같은 크기의, 0으로 채워진 배열 생성 / y,x,채널 : 3차원 행렬 image
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b  # ex) b=230
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    # upper_lip or lower_lip
    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]  # h,s만 바꿔주기(H:색조, S:채도)
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]  # h만 바꿔주기

    # 다시 BGR로 변환 / 3차원 changed
    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # ***boolean 조건문으로 배열 indexing*** : parsing결과 imaage(배열) 요소값이, hair면 17과 0으로 이루어짐. part(=17)과 다른 parsing부분(=17이 아닌 값)만 인덱싱해서 image(원본)의 인덱싱 결과값으로 대체!
    changed[parsing != part] = image[parsing != part]
    return changed


def color_to_BGR(color):
    b = color % 256
    color //= 256
    g = color % 256
    color //= 256
    r = color % 256
    return [b, g, r]


def makeup(path, color):
    b, g, r = color_to_BGR(color)
    filename = str(color) + '.jpg'

    # path : folder name(id)
    face = cv2.imread(os.path.join(path, 'face.jpg'))  # ori
    parsing = cv2.imread(os.path.join(path, 'parsing.jpg'))  # seg
    filled = face.copy()

    tar_color = np.zeros_like(face)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

    masked = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    filled[parsing == 12] = masked[parsing == 12]
    filled[parsing == 13] = masked[parsing == 13]
    blured = cv2.GaussianBlur(filled, (5, 5), 0)
    face[parsing == 12] = blured[parsing == 12]
    face[parsing == 13] = blured[parsing == 13]

    cv2.imwrite(os.path.join(path, filename), face)
