import csv
import os
from glob import glob
from posixpath import basename

COMPARATIVE_TABLE_PATH = '../data/csv/shelf_labels_new_old_comparative_table.csv'

def load_comparative_table(path):
    with open(COMPARATIVE_TABLE_PATH, newline='') as f:
        reader = csv.reader(f)
        comparative_table = {row[0]: row[1] for row in reader}
    return comparative_table

def rewrite_label(dir):
    comparative_table = load_comparative_table(COMPARATIVE_TABLE_PATH)
    
    paths = glob(dir + '*.jpg')
    for path in paths:
        label = [key for key in comparative_table.keys() if key in path]
        label = label[0] if label else label

        if label and label in comparative_table:
            index = path.find(label)
            label_new = comparative_table[label]
            new_path = path[:index] + label_new + path[index+len(label):]
            
            if not os.path.isfile(new_path):
                os.rename(path, new_path)

def rewrite_prefix(dir, prefix_old, prefix_new):
    paths = glob(dir + '*.jpg')

    for path in paths:
        new_path = dir + prefix_new + os.path.basename(path)[len(prefix_old):]
        if not os.path.isfile(new_path):
            os.rename(path, new_path)

def rewrite_suffix(dir, suffix_old, suffix_new):
    paths = glob(dir + '*.jpg')

    for path in paths:
        new_path = path[:-len(suffix_old)] + suffix_new
        if not os.path.isfile(new_path):
            os.rename(path, new_path)


if __name__ == '__main__':
    dir = 'C:/Users/b1018235/OneDrive - 公立はこだて未来大学/library/shelf/20210607/wide/'
    prefix_old = 'FUNlibrary_shoka'
    prefix_new = ''
    suffix_old = '_20210607.jpg'
    suffix_new = '.jpg'

    # rewrite_label(dir)
    rewrite_prefix(dir, prefix_old, prefix_new)
    rewrite_suffix(dir, suffix_old, suffix_new)