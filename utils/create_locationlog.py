import csv
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def composite(base, img, coord):
    alpha = Image.new('RGBA', base.size, (255, 255, 255, 0))
    alpha.paste(img, tuple(coord))
    return Image.alpha_composite(base, alpha)


def get_timestamp(start, offset):
    frames = np.sum(np.array([int(e)*60**(3-i) for i, e in enumerate(start.split(':'))])) + offset

    timestamp = []
    for _ in range(4):
        timestamp.append(str(frames%60).zfill(2))
        frames //= 60
    
    return (':').join(timestamp[::-1])


def main():
    # get a history
    with open('./csv/unsmoothed/ATARASHI_2021-07-12_0.csv', newline='') as f:
        unsmoothed_history = [row[1] for row in csv.reader(f)][1:]

    csv_path = './csv/smoothed/ATARASHI_2021-07-12_0.csv'
    with open(csv_path, newline='') as f:
        history = [row[0] for row in csv.reader(f)]


    # get coordinates
    with open('./csv/shelves_coordinate.csv', newline='') as f:
        rows = [row for row in csv.reader(f)][1:]
        SHELVES_XY = dict((row[0], [int(row[1]), int(row[2])]) for row in rows)
        SHELVES_ANGLE = dict((row[0], int(row[3])) for row in rows)    

    # 素材を読み込む
    MAP_IMG              = Image.open('./img/library_map.png').convert('RGBA')
    LOCATIOM_ICON_RED    = Image.open('./img/icon/navigation@1x_red.png').convert('RGBA')
    LOCATIOM_ICON_WHITE  = Image.open('./img/icon/navigation@1x_white.png').convert('RGBA')
    LOCATION_FONT        = ImageFont.truetype('./font/Mplus1-Medium.ttf', 60)
    TIMESTAMP_FONT       = ImageFont.truetype('./font/Digital7Mono-B1g5.ttf', 24)
    LOCATION_FONT_COLOR_RED   = (212, 20, 90)
    LOCATION_FONT_COLOR_WHITE = (255, 255, 255)
    TIMESTAMP_FONT_COLOR      = (255, 255, 255)

    # 動画インスタンスの作成
    FPS = 60
    out = cv2.VideoWriter('./video/log_'+os.path.basename(csv_path[:-4])+'.mp4',\
                             cv2.VideoWriter_fourcc('m','p','4','v'), FPS, tuple(MAP_IMG.size))

    for i, shlef in enumerate(history):
        frame = MAP_IMG.copy()

        # 現在位置（アイコン）の描画
        if shlef in SHELVES_XY:
            frame = composite(frame, LOCATIOM_ICON_RED.rotate(SHELVES_ANGLE[shlef]), SHELVES_XY[shlef])

        if unsmoothed_history[i] in SHELVES_XY:
            frame = composite(frame, LOCATIOM_ICON_WHITE.rotate(SHELVES_ANGLE[unsmoothed_history[i]]), SHELVES_XY[unsmoothed_history[i]])

        # 現在位置（テキスト）の描画
        ImageDraw.Draw(frame).text((220, 60), unsmoothed_history[i], font=LOCATION_FONT, fill=LOCATION_FONT_COLOR_WHITE)
        ImageDraw.Draw(frame).text((450, 60), shlef, font=LOCATION_FONT, fill=LOCATION_FONT_COLOR_RED)

        # 60fps部分
        for j in range(FPS//2):
            frame_temp = frame.copy()
            ImageDraw.Draw(frame_temp).text(xy=(5,5), text='2021-07-12 ' + get_timestamp('13:05:20',i*(FPS//2)+j),\
                                            font=TIMESTAMP_FONT, fill=TIMESTAMP_FONT_COLOR)
            out.write(cv2.cvtColor(np.array(frame_temp, dtype=np.uint8), cv2.COLOR_RGBA2BGR))

    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()