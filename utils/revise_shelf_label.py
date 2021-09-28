import csv
from glob import glob

COMPARATIVE_TABLE_PATH = '../data/csv/shelf_labels_new_old_comparative_table.csv'

def revise_shelf_labels(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        locations = [row for row in reader]

    with open(COMPARATIVE_TABLE_PATH, newline='') as f:
        reader = csv.reader(f)
        comparative_table = {row[0]: row[1] for row in reader}

    locations_revised = [locations[0]] + [[row[0], comparative_table[row[1]]] for row in locations[1:]]

    output_path = path[:-4] + '_revised.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(locations_revised)


if __name__ == '__main__':
    csv_dir = '../output/csv/'
    csv_paths = glob(csv_dir + '*.csv')

    for path in csv_paths:
        revise_shelf_labels(path)