# -*- coding: utf-8 -*-

import json
import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F


class MaskRCNN():
    """Mask R-CNN model pretrained by COCO dataset.

    COCO dataset classes
    0: 'unlabeled'
    1: 'person'
    2: 'bicycle'
    3: 'car'
    4: 'motorcycle'
    5: 'airplane'
    6: 'bus'
    7: 'train'
    8: 'truck'
    9: 'boat'
    10: 'traffic light' 
    11: 'fire hydrant' 
    12: 'street sign' 
    13: 'stop sign' 
    14: 'parking meter' 
    15: 'bench' 
    16: 'bird' 
    17: 'cat' 
    18: 'dog' 
    19: 'horse' 
    20: 'sheep' 
    21: 'cow' 
    22: 'elephant' 
    23: 'bear' 
    24: 'zebra' 
    25: 'giraffe' 
    26: 'hat' 
    27: 'backpack' 
    28: 'umbrella' 
    29: 'shoe' 
    30: 'eye glasses' 
    31: 'handbag' 
    32: 'tie' 
    33: 'suitcase' 
    34: 'frisbee' 
    35: 'skis' 
    36: 'snowboard' 
    37: 'sports ball' 
    38: 'kite' 
    39: 'baseball bat' 
    40: 'baseball glove' 
    41: 'skateboard' 
    42: 'surfboard' 
    43: 'tennis racket' 
    44: 'bottle' 
    45: 'plate' 
    46: 'wine glass' 
    47: 'cup' 
    48: 'fork' 
    49: 'knife' 
    50: 'spoon' 
    51: 'bowl' 
    52: 'banana' 
    53: 'apple' 
    54: 'sandwich' 
    55: 'orange' 
    56: 'broccoli' 
    57: 'carrot' 
    58: 'hot dog' 
    59: 'pizza' 
    60: 'donut' 
    61: 'cake' 
    62: 'chair' 
    63: 'couch' 
    64: 'potted plant' 
    65: 'bed' 
    66: 'mirror' 
    67: 'dining table' 
    68: 'window' 
    69: 'desk' 
    70: 'toilet' 
    71: 'door' 
    72: 'tv' 
    73: 'laptop' 
    74: 'mouse' 
    75: 'remote' 
    76: 'keyboard' 
    77: 'cell phone' 
    78: 'microwave' 
    79: 'oven' 
    80: 'toaster' 
    81: 'sink' 
    82: 'refrigerator' 
    83: 'blender' 
    84: 'book' 
    85: 'clock' 
    86: 'vase' 
    87: 'scissors' 
    88: 'teddy bear' 
    89: 'hair drier' 
    90: 'toothbrush'

    Attributes:
        model (nn.Module): Mask R-CNN model.
    """

    def __init__(self):
        """Freeze the whole network just for inference and upload the model to GPU memory for CUDA use."""
        
        torch.set_grad_enabled(False)
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.eval().cuda()
        
        self.COCO_CLASSES = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 
        self.coco_class_to_extract = ['book']


    def print_coco_labels(self):
        for i, label in enumerate(self.COCO_LABELS):
            print(f'index: {i}, label: {label}')


    def compute_mask_heatmap(self, output, score_threshold=0.5):
        """Concatenate all regions of "BOOK" class.

        Args:
            output (torch.Tensor): [description]
            score_threshold (float): Default value is followed by "TorchvisionMaskRCNN.ipynb".

        Returns:
            numpy.ndarray : [description]
        """

        mask_sum = torch.zeros_like(output['masks'][0])

        for label, score, mask in zip(output['labels'], output['scores'], output['masks']):
            if score > score_threshold and self.COCO_CLASSES[label] in self.coco_class_to_extract:
                mask_sum = torch.max(mask, mask_sum)
    
        mask_sum = mask_sum.squeeze(0).cpu().numpy()
        return mask_sum
    

    def masking(self, input_image):
        """Masking the input image

        Args:
            image (ndarray): input image (RGB).

        Returns:
            ndarray: image filled with #000000 except 'book' class region.
        """

        input_image_tensor = F.to_tensor(input_image).cuda()
        output = self.model([input_image_tensor])[0] #[0] is necessary (I forgot why but I assume self.model has 2 dimensions.)
        
        mask            = self.compute_mask_heatmap(output)
        _, binary_mask  = cv2.threshold(mask, 0, 255, type=cv2.THRESH_BINARY)
        binary_mask_gbr = np.tile(binary_mask, (3, 1, 1)).transpose(1, 2, 0).astype(np.uint8)
        
        masked_image = cv2.bitwise_and(input_image, binary_mask_gbr)

        return masked_image


with open('/src/shelf_matching/data/settings.json', 'r') as f:
    SETTINGS = json.load(f)

RESIZE_RATIO = SETTINGS['RESIZE_RATIO']


def preprocess(image, mask_model):
    w = int(RESIZE_RATIO*image.shape[1])
    h = int(RESIZE_RATIO*image.shape[0])

    image = cv2.resize(image, (w,h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = mask_model.masking(image)
    # ここに前処理を追加していく

    preprocessed_image = image
    return preprocessed_image