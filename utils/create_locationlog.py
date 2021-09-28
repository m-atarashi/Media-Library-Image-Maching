# csv読み込み
# forで一フレームずつ描画
# マップ画像読み込み
# 座標計算
# 現在地アイコン描画
# 軌跡アイコン描画
# 動画に書き込み
# 繰り返す

import csv
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 書架の履歴を規定したcsvファイルを読み込む
csv_dir = '../output/csv/'
csv_paths = glob(csv_dir + '*.csv')
csv_path = csv_paths[3]

with open(csv_path, newline='') as f:
    locations = [row[1] for row in csv.reader(f)][1:]


# 画像素材を読み込む
map_img_path = '../data/img/map/library_map.png'
map_img = Image.open(map_img_path).convert('RGBA')

location_icon_path = '../data/img/icon/navigation@1x.png'
location_icon = Image.open(location_icon_path).convert('RGBA')

location_icon_l_path = '../data/img/icon/outline_donut_large_black_48dp.png'
location_icon_l = Image.open(location_icon_l_path).convert('RGBA')


# 座標を規定したcsvファイルを読み込む
coords_path = '../data/csv/shelves_coordinate.csv'
with open(coords_path, newline='') as f:
    data = [row for row in csv.reader(f)][1:]
    keys = [row[0] for row in data]
    values = [(int(row[1]), int(row[2])) for row in data]
    coords = dict(zip(keys, values))
    angles = dict(zip(keys, [int(row[3]) for row in data]))    


# 動画インスタンスの作成
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 30
out = cv2.VideoWriter('../video/log_' + os.path.basename(csv_path[:-4]) + '.mp4', fourcc, fps, (1800, 510))


# 位置情報（2fps）の
for i, location in enumerate(locations):

    frame = map_img.copy()

    # 現在位置の描画
    if location in coords:
        coord = coords[location]
        angle = angles[location]

        alpha = Image.new('RGBA', frame.size, (255, 255, 255, 0))
        alpha.paste(location_icon.rotate(angle), coord)
        frame = Image.alpha_composite(frame, alpha)

    # 位置情報の表示（文字）
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype('../data/font/Mplus1-Medium.ttf', 60)
    draw.text((304, 60), location, font=font)

    # フレーム画像の生成
    for j in range(fps//2):
        alpha = Image.new('RGBA', frame.size, (255, 255, 255, 0))
        alpha.paste(location_icon_l.rotate(-360*(fps//2*(i%2)+j)/fps), (207, 57))
        frame_temp = Image.alpha_composite(frame, alpha)

        out.write(cv2.cvtColor(np.array(frame_temp, dtype=np.uint8), cv2.COLOR_RGBA2BGR))

out.release()
cv2.destroyAllWindows()