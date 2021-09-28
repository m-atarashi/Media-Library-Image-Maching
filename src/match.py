# -*- coding: utf-8 -*-
import json
import os
from collections import Counter
from operator import itemgetter
from time import time

import cv2
import faiss
import numpy as np
import pandas as pd
from scipy import stats

import database
import describe
import nns
import preprocessing
from utils import utils


with open('/src/shelf_matching/data/user_settings.json', 'r') as f:
    USER_SETTINGS = json.load(f)

USERNAME      = USER_SETTINGS['USERNAME']
SHOOTING_DATE = USER_SETTINGS['SHOOTING_DATE']
VIDEO_ID      = USER_SETTINGS['VIDEO_ID']

with open('/src/shelf_matching/data/settings.json', 'r') as f:
    SETTINGS = json.load(f)

DETECTOR_NAME                   = SETTINGS['DETECTOR']
HESSIAN_THRESHOLD               = SETTINGS['HESSIAN_THRESHOLD']
DESCRIPTION_DISTANCE_THRESHOLDS = SETTINGS['DESCRIPTION_DISTANCE_THRESHOLDS'][DETECTOR_NAME]
kNN_RATIO                       = SETTINGS['kNN_RATIOS'][DETECTOR_NAME]
LOSS_WEIGHTS                    = SETTINGS['LOSS_WEIGHTS']
DISPLAY_COLS_NUM                = SETTINGS['DISPLAY_COLS_NUM']
SHELF_LABELS                    = [os.path.basename(path)[:-4] for path in utils.glob_extensions(['jpg', 'JPG', 'png', 'PNG'], SETTINGS['SHELF_DIR'])]


def match_over_all_shelves(pov_images_dir, shelf_descriptors):
    """compute matchings over all input images

    Args
        ShelfMatcher (ShelfMatcher): [description]
        shelf_descriptors (List[ndarray]): list of pre-calculated descriptors of shelf images
        start_index (int, optional): [description]. Defaults to 0.
        n (int, optional): [description]. Defaults to None.
    """

    file_extensions = ['jpg', 'JPG', 'png', 'PNG']
    pov_image_paths = utils.glob_extensions(file_extensions, pov_images_dir)
    mask_model = preprocessing.MaskRCNN()
    gpu_resources = faiss.StandardGpuResources()

    inputs  = []
    answers = []

    start_secs = time()
    for pov_image_path in pov_image_paths:
        try:
            answer = match(pov_image_path, shelf_descriptors, mask_model, gpu_resources)
        except:
            answer = 'None'
        finally:
            inputs.append(os.path.basename(pov_image_path)[:-4])
            answers.append(answer)

    elapsed_secs = time() - start_secs
    print('Mean processing time:', elapsed_secs/len(pov_image_paths))

    return inputs, answers


def match(pov_image_path, shelf_descriptors, mask_model, gpu_resources):
    """1フレームの一人称画像にどの書架が映っているかを計算する
    1. 画像の読み込み
    2. 前処理
    3. 特徴量の距離の計算
    4. データフレームの作成

    Args:
        pov_image_path (String): 一人称画像のパス
        shelf_descriptors (ndarray): 事前に計算したすべての書架の特徴量

    Returns:
        String: 書架のラベル
    """

    pov_image          = cv2.imread(pov_image_path, mask_model)
    preprocessed_image = preprocessing.preprocess(pov_image)
    distances          = nns_over_all_shelves(preprocessed_image, shelf_descriptors, gpu_resources)
    df                 = compute_result_table(distances)
    answer             = SHELF_LABELS[df.index.values[0]]
    return answer


def nns_over_all_shelves(pov_image, shelf_descriptors, gpu_resources):
    detector = describe.detectors(DETECTOR_NAME, HESSIAN_THRESHOLD)
    _, pov_descriptor = detector.detectAndCompute(pov_image, None)

    distances = []
    is_binary = describe.use_hamming_distance(DETECTOR_NAME)

    for shelf_descriptor in shelf_descriptors:
        D, _ = nns.faiss_knnmatch(gpu_resources, shelf_descriptor, pov_descriptor, is_binary)
        indices = (D[:, 0] < kNN_RATIO*D[:, 1]) & (D[:, 0] < DESCRIPTION_DISTANCE_THRESHOLDS)
        distances.append(D[indices, 0])
    
    distances = np.array(distances, dtype = object)
    return distances


def compute_result_table(distances):
    """Calculate losses of each matching between a pov image and every shelf.
    Args:
        distances (ndarray): results of matching between a pov image and every shelf
    
    Returns:
        df (DataFrame): dataFrame containing columns of shelf label, the count of each matching points, mean distance between descriptors and loss value
    """

    # delete None
    indices      = [i for i, d in enumerate(distances) if any(d)]
    distances    = distances[indices]

    shelf_labels = itemgetter(*indices)(SHELF_LABELS)
    counts       = [len(d) for d in distances]
    distances    = [np.mean(d) for d in distances]
    losses       = LOSS_WEIGHTS['DISTANCE_WEIGTH'] * stats.zscore(distances) - \
                   LOSS_WEIGHTS['COUNT_WEIGHT']    * stats.zscore(counts)

    df = pd.DataFrame({'Shelf'   : shelf_labels,
                       'Count'   : counts,
                       'Distance': distances,
                       'Loss'    : losses},
                       columns = ['Shelf', 'Count', 'Distance', 'Loss'])
    df = df.sort_values('Loss')
    print(df.iloc[:DISPLAY_COLS_NUM, :].to_string(index=False))

    return df


def write_csv(dir, inputs, answers):
    df   = pd.DataFrame({'Input': inputs, 'Answers': answers}, columns=['Input', 'Answers'])
    path = f'{dir}/{USERNAME}_{SHOOTING_DATE}_{VIDEO_ID}.csv'
    df.to_csv(path, index=False)


def write_totalled_result(dir, answers):
    data = dict(Counter(answers))
    path = f'{dir}/{USERNAME}_{SHOOTING_DATE}_{VIDEO_ID}_totalled_result.json'

    with open(path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_setting(dir):
    data = {
        'username'          : USERNAME,
        'date'              : SHOOTING_DATE,
        'video_id'          : VIDEO_ID,
        'detector'          : DETECTOR_NAME,
        'hessian_threshold' : HESSIAN_THRESHOLD,
        'distance_threshold': DESCRIPTION_DISTANCE_THRESHOLDS,
        'k-NN ratio'        : kNN_RATIO,
        'loss_weight'       : LOSS_WEIGHTS
    }
    path = f'{dir}/{USERNAME}_{SHOOTING_DATE}_{VIDEO_ID}_setting.json'
    with open(path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def insert_logdata_into_database(answers):
    shelf_counts = database.shelf_counts_json(answers)

    sql = f'INSERT INTO logdata (userid, log) VALUES (%s, %s)'
    values = (USERNAME, shelf_counts)
    database.insert_logdata_into_database('localhost', 'root', 'password', 'peimura_game', sql, values)