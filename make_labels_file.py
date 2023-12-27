import csv
import os
import config


def make_labels_file():
    if os.path.isfile(config.LABELS_FILE):
        print(f'Labels file {config.LABELS_FILE} exists')
    else:
        fields = ['img', 'lat', 'lon']
        rows = []

        for i in os.listdir(config.IMG_DIR):
            if i.endswith('.jpg'):
                lat, lon = i.split('@')[0].split('_')
                # print(i, lat, lon)
                rows.append([i, lat, lon])

        with open(config.LABELS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)

        print(f'Created {config.LABELS_FILE}')


if __name__ == '__main__':
    make_labels_file()