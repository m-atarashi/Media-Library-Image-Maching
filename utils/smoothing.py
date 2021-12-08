import csv
import numpy as np
from numpy.core.fromnumeric import argmin
import pandas as pd


def smoothing_by_cordinate(history):
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
    method = 'WMA'
    if method == 'WMA':
        smoothed_history_x = np.convolve(history_y_interpolated, np.ones(11)/11, mode='same')
        smoothed_history_y = np.convolve(history_y_interpolated, np.ones(11)/11, mode='same')
    else:
        smoothed_history_x = history_x_interpolated.ewm(com=len(history_x)-1, adjust=False).mean().values
        smoothed_history_y = history_y_interpolated.ewm(com=len(history_y)-1, adjust=False).mean().values
    
    smoothed_history = np.stack([smoothed_history_x, smoothed_history_y], axis=1)
    print([np.linalg.norm(values-e, axis=1).argmin() for e in smoothed_history][-10:])

    # quantize the coordinates
    indices = [np.linalg.norm(values - e, axis=1).argmin() for e in smoothed_history]
    smoothed_history = np.array(keys)[indices]

    return smoothed_history


def main():
    path = './csv/unsmoothed/ATARASHI_2021-05-18_0.csv'
    with open(path, "r") as f:
        reader  = csv.reader(f)
        history = [row[1] for row in reader][1:]

    smoothed_history = smoothing_by_cordinate(history)

    path = './csv/smoothed/ATARASHI_2021-05-18_0.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[e] for e in smoothed_history])


if __name__ == '__main__':
    main()