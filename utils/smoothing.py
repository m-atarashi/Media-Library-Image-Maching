import csv
import os
from glob import glob

import numpy as np
import pandas as pd
import scipy.signal


def smoothing_by_cordinate(history, method='WMA', window_size=5):
    coordinates_path = './csv/shelves_coordinate.csv'
    with open(coordinates_path, 'r') as f:
        reader = csv.reader(f)
        data   = [row for row in reader][1:]

        keys   = [row[0] for row in data]
        values = [[int(row[1]), int(row[2])] for row in data]
        coords = dict(zip(keys, values))
    
    # get coordinaes
    history_coordinate = [[np.nan, np.nan] if e == "None" else coords[e] for e in history]
    history_coordinate = np.array(history_coordinate)
    history_x = history_coordinate[:, 0]
    history_y = history_coordinate[:, 1]

    # inter@plation of Nan value
    history_x_interpolated = pd.Series(history_x).interpolate(limit_direction='both')
    history_y_interpolated = pd.Series(history_y).interpolate(limit_direction='both')

    # smoothing
    if method == 'SMA':
        smoothed_history_x = np.convolve(history_x_interpolated, np.ones(window_size)/window_size, mode='same')
        smoothed_history_y = np.convolve(history_y_interpolated, np.ones(window_size)/window_size, mode='same')
    elif method == 'WMA':
        weight = lambda n: np.arange(1, n+1)[::-1]/np.sum(np.arange(1, n+1))
        smoothed_history_x = np.convolve(history_x_interpolated, weight(window_size), mode='same')
        smoothed_history_y = np.convolve(history_y_interpolated, weight(window_size), mode='same')
    elif method == 'EMA':
        smoothed_history_x = history_x_interpolated.ewm(alpha=1/window_size, adjust=False).mean().values
        smoothed_history_y = history_y_interpolated.ewm(alpha=1/window_size, adjust=False).mean().values
    elif method == 'SG':
        smoothed_history_x = scipy.signal.savgol_filter(history_x_interpolated, window_size, 2, deriv=0)
        smoothed_history_y = scipy.signal.savgol_filter(history_y_interpolated, window_size, 2, deriv=0)

    smoothed_history_coordinate = np.stack([smoothed_history_x, smoothed_history_y], axis=1)

    # quantize the coordinates
    indices = [np.argmin(np.linalg.norm(values - e, axis=1)) for e in smoothed_history_coordinate]
    smoothed_history = np.array(keys)[indices]

    return smoothed_history


def main():
    paths = glob('./csv/unsmoothed/*.csv')
    for path in paths:
        with open(path, "r") as f:
            reader  = csv.reader(f)
            history = [row[1] for row in reader][1:]

        smoothed_history = smoothing_by_cordinate(history, method='WMA', window_size=5)

        path = './csv/smoothed/' + os.path.basename(path)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[e] for e in smoothed_history])


if __name__ == '__main__':
    main()
