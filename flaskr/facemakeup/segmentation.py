import numpy as np
import cv2 as cv2
import os
def segmentation(path):
    img = cv2.imread(os.path.join(path, 'face.jpg'))
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()