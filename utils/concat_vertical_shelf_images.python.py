import os
import glob
import numpy as np
import cv2

dir = 'C:/Users/b1018235/Documents/Main/2x/'

paths = glob.glob(dir + '*.jpg')

for i in range(0, 2, 2):
    image1 = cv2.imread(paths[i])
    image2 = cv2.imread(paths[i+1])
    image_concat = np.concatenate([np.array(image1), np.array(image2)], axis=1)
    cv2.imwrite('C:/Users/b1018235/Documents/Main/2x/concat/a'  + '.jpg', image_concat)
