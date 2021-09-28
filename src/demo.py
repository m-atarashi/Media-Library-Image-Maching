import json
import os
import warnings

import numpy as np
import match


with open('/src/shelf_matching/data/settings.json', 'r') as f:
    SETTINGS = json.load(f)

with open('/src/shelf_matching/data/user_settings.json', 'r') as f:
    USER_SETTINGS = json.load(f)

POV_IMAGES_DIR = USER_SETTINGS['POV_DIR']
OUTPUT_DIR     = SETTINGS['OUTPUT_DIR']
PARAM_PATH     = SETTINGS['PARAM_PATH']


if __name__ == '__main__':
    warnings.simplefilter('ignore')

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    shelf_descriptors = np.load(PARAM_PATH, allow_pickle=True)
    inputs, answers = match.match_over_all_shelves(POV_IMAGES_DIR, shelf_descriptors)

    match.write_csv(OUTPUT_DIR, inputs, answers)
    match.write_totalled_result(OUTPUT_DIR, answers)
    match.write_setting(OUTPUT_DIR)
    match.insert_logdata_into_database(answers)