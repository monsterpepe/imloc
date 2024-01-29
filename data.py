import numpy as np
import os
import pandas as pd
import time

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

import config
from make_labels_file import make_labels_file


class ImageDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, label_transform=None):
        self.labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        # numpy image: H x W x C
        # torch image: C x H x W
        try:
            img = read_image(img_file, ImageReadMode.RGB)
        except Exception as e:
            raise Exception(img_file)

        labels = self.labels.iloc[idx, 1:]
        labels = np.array(labels, dtype=float)
        if self.transform:
            img = self.transform(img)
        if self.label_transform:
            labels = self.label_transform(labels)
        return img, labels


def linear_scale(labels):
    lat = labels[0]
    lng = labels[1]
    y1 = (lat-config.MIN_LAT) / (config.MAX_LAT-config.MIN_LAT)
    y2 = (lng-config.MIN_LNG) / (config.MAX_LNG-config.MIN_LNG)
    return torch.as_tensor([y1, y2])


def get_transform(preprocess=None):
    transform = v2.Compose([
        v2.Resize(config.IMG_SIZE, antialias=True), # resize shortest edge to config.IMG_SIZE
        # v2.CenterCrop(config.IMG_SIZE), # random vs center crop
        v2.RandomCrop(config.IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    if preprocess:
        transform = v2.Compose([transform, preprocess])
    label_transform = v2.Compose([
            linear_scale,
            v2.ToDtype(torch.float32),
    ])
    return transform, label_transform


def get_loaders(preprocess=None):
    make_labels_file()
    transform, label_transform = get_transform(preprocess)

    dataset = ImageDataset(
        labels_file=config.LABELS_FILE,
        img_dir=config.IMG_DIR,
        transform=transform,
        label_transform=label_transform,
    )
    print(f'Created dataset: {len(dataset)}')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    g = torch.Generator()
    g.manual_seed(42)

    train_dataset, val_dataset,  test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=g)
    print('Train, val, test:', len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
        num_workers=4,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
        num_workers=4,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE,
        num_workers=4,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST,
    )
    print('Created train, val, test loaders')

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    start = time.time()
    train_loader, val_loader, test_loader = get_loaders()
    print('Loading train data...')
    for batch, (X, y) in enumerate(train_loader):
        if not batch % 100:
            print(f'Train {batch} batches @ {time.time()-start}')
    print('Loading val data...')
    for batch, (X, y) in enumerate(val_loader):
        if not batch % 100:
            print(f'Val {batch} batches @ {time.time()-start}')
    print('Loading test data...')
    for batch, (X, y) in enumerate(test_loader):
        if not batch % 100:
            print(f'Test {batch} batches @ {time.time()-start}')
    print('All data loaded')
