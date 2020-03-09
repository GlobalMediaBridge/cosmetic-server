import numpy as np
import cv2 as cv2
import os
from flaskr.facemakeup.test import evaluate
from flaskr.facemakeup.makeup import makeup

def segmentation(path):
    #path : 폴더명(id)
    current_dir = os.path.dirname(os.path.abspath(__file__)) #절대경로
    image_path = os.path.join(path, 'face.jpg')
    cp = os.path.join(current_dir, 'cp/79999_iter.pth') #상대경로->절대경로로

    img = cv2.imread(image_path) #우선 이미지 읽어들이기
    ori = img.copy() #original
    parsing = evaluate(image_path, cp) # test.py의 evaluate() 함수. (512, 512)의 image (parsing) return.
    parsing = cv2.resize(parsing, img.shape[0:2], interpolation=cv2.INTER_NEAREST) #이미지 사이즈 조정. 픽셀사이 값 보간. / image.shape[] : [Y축, X축, 채널수] 순서 중 0,1번지 2개만 가져옴
    
    parsing_img_path = os.path.join(path, 'parsing.jpg')
    cv2.imwrite(parsing_img_path, parsing)

    makeup_img = makeup(image_path, parsing_img_path)
    makeup_img_path = os.path.join(path, 'makeup.jpg')
    cv2.imwrite(makeup_img_path, makeup_img)

    #다 완성되면 이렇게 확인하는 코드는 삭제하기! 
    cv2.imshow('ori', cv2.resize(ori, (512,512))) #original img
    cv2.imshow('seg', cv2.resize(parsing.astype('uint8'), (512,512))) #segmentation img
    #print(np.shape(parsing), type(parsing)) #(960, 960), <class 'numpy.ndarray'>
    cv2.imshow('makeup', cv2.resize(makeup_img.astype('uint8'), (512,512)))
    
    cv2.waitKey()
    cv2.destroyAllWindows()