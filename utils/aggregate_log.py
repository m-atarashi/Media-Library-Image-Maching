import os
import glob
import csv
import numpy as np
from collections import Counter, defaultdict

dir_log = '../output/csv/'
paths = glob.glob(dir_log + '*.csv')

path_log = paths[1]
with open(path_log, newline='') as f:
    log = [row[1] for row in csv.reader(f)][1:]


path_comparative_table = '../data/csv/comparative_table_for_aggregate.csv'
with open(path_comparative_table, newline='') as f:
    reader = list(csv.reader(f))
    keys = [row[0] for row in reader]
    values = [row[1] for row in reader]
    old_new_table = dict(zip(keys, values))


counts = defaultdict(int)
counter = Counter(log)
for key, value in old_new_table.items():
    counts[value] += counter[key] if key in counter else 0
counts_li = zip(*counts.items())

with open('../data/csv/count_aggregated.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(counts_li)