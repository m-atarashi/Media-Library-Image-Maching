import csv
import glob
import json

path = "C:/Users/b1018235/OneDrive - 公立はこだて未来大学/library/data/csv/count_aggregated.csv"

with open(path, "r") as f:
    reader = csv.reader(f)
    row = list(reader)
    keys, values = row[:2]
    json_data = dict(zip(keys, values))
    print(json_data)

with open("log.json") 