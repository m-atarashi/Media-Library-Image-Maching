import csv
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    # get a history
    csv_paths = glob('./csv/smoothed/*.csv')
    csv_path = csv_paths[3]

    with open(csv_path, newline='') as f:
        locations = [row[0] for row in csv.reader(f)]

    # 画像素材を読み込む
    map_img = Image.open('./img/library_map.png').convert('RGBA')
    location_icon = Image.open('./img/icon/navigation@1x.png').convert('RGBA')
    loding_icon = Image.open('./img/icon/outline_donut_large_black_48dp.png').convert('RGBA')

    # get coordinates
    with open('./csv/shelves_coordinate.csv', newline='') as f:
        rows = [row for row in csv.reader(f)][1:]
        coords = dict((row[0], [row[1], row[2]]) for row in rows)
        angles = dict((row[0], int(row[3])) for row in rows)    

    return 1

    # 動画インスタンスの作成
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 30
    out = cv2.VideoWriter('./video/log_' + os.path.basename(csv_path[:-4]) + '.mp4', fourcc, fps, (1800, 510))

    for i, location in enumerate(locations):
        frame = map_img.copy()

        # 現在位置の描画
        if location in coords:
            alpha = Image.new('RGBA', frame.size, (255, 255, 255, 0))
            alpha.paste(location_icon.rotate(angles[location]), coords[location])
            frame = Image.alpha_composite(frame, alpha)

        # 位置情報の表示（文字）
        ImageDraw.Draw(frame).text((304, 60), location, font=ImageFont.truetype('./font/Mplus1-Medium.ttf', 60))

        # フレーム画像の生成
        for j in range(fps//2):
            alpha = Image.new('RGBA', frame.size, (255, 255, 255, 0))
            alpha.paste(loding_icon.rotate(-360*(fps//2*(i%2)+j)/fps), (207, 57))
            frame_temp = Image.alpha_composite(frame, alpha)

            out.write(cv2.cvtColor(np.array(frame_temp, dtype=np.uint8), cv2.COLOR_RGBA2BGR))

    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()