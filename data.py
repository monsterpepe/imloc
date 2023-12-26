import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

import torch
# from torch import multiprocessing
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torchvision.io import read_image

import config
from make_labels_file import make_labels_file


class ImageDataset(Dataset): # ImageFolder???
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        # numpy image: H x W x C
        # torch image: C x H x W
        try:
            img = read_image(img_path)
        except Exception as e:
            raise Exception(img_path)

        labels = self.labels.iloc[idx, 1:]
        labels = np.array(labels, dtype=float)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return img, labels


def linear_scale(labels):
    lat = labels[0]
    lng = labels[1]
    y1 = (lat-config.MIN_LAT) / (config.MAX_LAT-config.MIN_LAT)
    y2 = (lng-config.MIN_LNG) / (config.MAX_LNG-config.MIN_LNG)
    return torch.tensor([y1, y2])


def make_dataloaders():
    make_labels_file()

    transform = v2.Compose([
        v2.Resize(config.IMG_SIZE, antialias=True), # shortest edge is config.IMG_SIZE
        # v2.CenterCrop(config.IMG_SIZE), # random vs center crop
        v2.RandomCrop(config.IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    target_transform = v2.Compose([
            linear_scale,
            v2.ToDtype(torch.float32),
    ])

    dataset = ImageDataset(
        labels_file=config.LABELS_FILE,
        img_dir=config.IMG_DIR,
        transform=transform,
        target_transform=target_transform,
    )

    print(f'Created dataset: {len(dataset)}')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset,  test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    print('Train, val, test:', len(train_dataset), len(val_dataset), len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=config.SHUFFLE,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=config.SHUFFLE,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=config.SHUFFLE,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
    )

    print('Created train, val, test dataloaders')

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    start = time.time()
    # multiprocessing.set_start_method('spawn')

    train_dataloader, val_dataloader, test_dataloader = make_dataloaders()
    # for batch, (X, y) in enumerate(train_dataloader):
    #     print(f'Batch: {batch}')
    #     print(f'X: {X.shape}')
    #     print(f'y: {y.shape}')

    #     img = X[0]
    #     plt.imshow(img.permute(1, 2, 0))
    #     plt.show()

    #     break
    print('Loading train data...')
    for batch, (X, y) in enumerate(train_dataloader):
        if not batch % 100:
            print(f'Train n_batch@{time.time()-start}: {batch}')
    print('Loading val data...')
    for batch, (X, y) in enumerate(val_dataloader):
        if not batch % 100:
            print(f'Val n_batch@{time.time()-start}: {batch}')
    print('Loading test data...')
    for batch, (X, y) in enumerate(test_dataloader):
        if not batch % 100:
            print(f'Test n_batch@{time.time()-start}: {batch}')
    print('Data loading passed')