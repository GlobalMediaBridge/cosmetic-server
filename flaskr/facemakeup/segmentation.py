import numpy as np
import cv2 as cv2
import os
from flaskr.facemakeup.test import evaluate
def segmentation(path):
    '''
    img = cv2.imread(os.path.join(path, 'face.jpg'))
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    '''
    # dictionary형. table['hair']=17로 사용
    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }
    '''

    current_dir = os.path.dirname(os.path.abspath(__file__)) #절대경로
    image_path = os.path.join(path, 'face.jpg')
    cp = os.path.join(current_dir, 'cp/79999_iter.pth') #상대경로->절대경로로

    img = cv2.imread(image_path) #우선 이미지 읽어들이기
    ori = img.copy() #original
    parsing = evaluate(image_path, cp) # test.py의 evaluate() 함수. (512, 512)의 image (parsing) return.
    parsing = cv2.resize(parsing, img.shape[0:2], interpolation=cv2.INTER_NEAREST) #이미지 사이즈 조정. 픽셀사이 값 보간. / image.shape[] : [Y축, X축, 채널수] 순서 중 0,1번지 2개만 가져옴

    '''
    parts = [table['hair'], table['upper_lip'], table['lower_lip']] #[17, 12, 13]

    colors = [[17, 17, 63], [42.75, 63.84, 208.98], [42.75, 63.84, 208.98]] #[b,g,r] 순서, [[hair??],[upper_lip],[lower_lip]]

    # 두 변수 part라는 index는 parts를, color는 colors를 동시에 for문을 돌음
    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color) #hair함수에 들어가는 인자 image는 원래 원본이미지 / 한 부분, 한 색상(bgr)씩 들어감 => 3번 돌면 image 완성

    cv2.imshow('ori', cv2.resize(ori, (512, 512)))    # 원본이미지 
    cv2.imshow('color', cv2.resize(image, (512, 512)))  # makeup이미지
    '''

    cv2.imshow('ori', cv2.resize(ori, (512,512))) #original
    #print(np.shape(parsing), type(parsing)) #(960, 960), <class 'numpy.ndarray'>
    cv2.imshow('seg', parsing.astype('uint8')) #segmentation

    cv2.waitKey()
    cv2.destroyAllWindows()