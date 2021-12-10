import csv
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    # get a history
    csv_path = glob('./csv/smoothed/*.csv')[3]
    with open(csv_path, newline='') as f:
        history = [row[0] for row in csv.reader(f)]

    with open('./csv/unsmoothed/ATARASHI_2021-07-12_0.csv', newline='') as f:
        unsmoothed_history = [row[0] for row in csv.reader(f)]

    # get coordinates
    with open('./csv/shelves_coordinate.csv', newline='') as f:
        rows = [row for row in csv.reader(f)][1:]
        coords = dict((row[0], [int(row[1]), int(row[2])]) for row in rows)
        angles = dict((row[0], int(row[3])) for row in rows)    

    # 画像素材を読み込む
    map_img = Image.open('./img/library_map.png').convert('RGBA')
    loding_icon = Image.open('./img/icon/outline_donut_large_black_48dp.png').convert('RGBA')
    location_icon_red = Image.open('./img/icon/navigation@1x_red.png').convert('RGBA')
    location_icon_green = Image.open('./img/icon/navigation@1x_green.png').convert('RGBA')


    # 動画インスタンスの作成
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 30
    out = cv2.VideoWriter('./video/log_' + os.path.basename(csv_path[:-4]) + '.mp4', fourcc, fps, (1800, 510))

    for i, shlef in enumerate(history):
        frame = map_img.copy()

        # 現在位置（アイコン）の描画
        if shlef in coords:
            frame = composite(frame, location_icon_red.rotate(angles[shlef]), coords[shlef])

        if unsmoothed_history[i] in coords:
            frame = composite(frame, location_icon_green.rotate(angles[unsmoothed_history[i]]), coords[unsmoothed_history[i]])

        # 現在位置（テキスト）の描画
        ImageDraw.Draw(frame).text((304, 60), shlef, font=ImageFont.truetype('./font/Mplus1-Medium.ttf', 60))

        # 30fps部分
        for j in range(fps//2):
            frame = composite(frame, loding_icon.rotate(-360*(fps//2*(i%2)+j)/fps), (207, 57))
            out.write(cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGBA2BGR))

    out.release()
    cv2.destroyAllWindows()


def composite(base, img, coord):
    alpha = Image.new('RGBA', base.size, (255, 255, 255, 0))
    alpha.paste(img, tuple(coord))
    return Image.alpha_composite(base, alpha)


if __name__ == '__main__':
    main()