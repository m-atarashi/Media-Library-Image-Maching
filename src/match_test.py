import json

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
    shelf_descriptors = np.load(PARAM_PATH, allow_pickle=True)
    pov_image_path = "../pov/test/20210607_10150.jpg"
    answer = match.match(pov_image_path, shelf_descriptors)
