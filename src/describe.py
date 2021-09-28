# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob

import cv2
import numpy as np


def detectors(detector_name, hessian_threshold=None):
    if detector_name == 'AKAZE':
        return cv2.AKAZE_create()

    elif detector_name == 'SIFT':
        return cv2.xfeatures2d.SIFT_create()

    elif detector_name == 'SURF':
        return cv2.xfeatures2d.SURF_create(hessian_threshold)

    elif detector_name == 'SURF_CUDA':
        return cv2.cuda.SURF_CUDA_create(hessian_threshold)

    elif detector_name == 'ORB':
        return cv2.ORB_create()


def use_hamming_distance(detector_name):
    dict = defaultdict(bool)
    dict['AKAZE'] = True
    dict['ORB']   = True
    return dict


# Methods
# compute(): Compute each descriptor of input images
# - Parameters
#       detector: detector of keypoints
#       dir: directory of images for description
#       format: file extension (default: .jpg)
#       scale: resize scale (default: not resize)
# - Returns: 
#       a list of descriptors
def compute(detector, dir, format='jpg', scale=1):
    paths = sorted(glob(dir + '*.' + format))
    descriptors = []

    for path in paths:
        image = cv2.imread(path)
        w = int(scale*image.shape[1])
        h = int(scale*image.shape[0])

        if scale != 1:
            image = cv2.resize(image, (w,h))

        _, descriptor = detector.detectAndCompute(image, None)
        descriptors.append(descriptor)
    return descriptors


# save(): Save descriptors as .npy
# - Parameters
#       descriptors: list of descriptors to save
#       path: file name (except for the extention '.npy')
def save(path, descriptors):
    descriptors = np.array(descriptors)
    np.save(path, descriptors)

    print('Descriptors saved successfully.')