import os
import time

import torch

from data import make_dataloaders
from net import ResNet50Pretrained, ResNet50_Weights, resnet50
import config

start = time.time()

LOG_FILE = os.path.join(config.MODEL_DIR, f'log.txt')


def train(dataloader, model, loss_fn, optimizer, preprocess=None, device='cuda'):
    print('Training...')
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if preprocess:
            X = preprocess(X)

        # Compute prediction error (forward pass)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backpropagation (backward pass)
        optimizer.zero_grad(set_to_none=True) # ~2s time reduction
        loss.backward()
        optimizer.step() # update weights

        loss_log = f'Train loss ({batch+1}@{round(time.time()-start)}): {loss.item()}'
        print(loss_log)
        with open(LOG_FILE, 'a') as f:
            f.write(f'{loss_log}\n')


def validate(dataloader, model, loss_fn, preprocess=None, device='cuda'):
    print('Testing...')
    model.eval()
    avg_loss = 0
    with torch.no_grad(): # disable gradient calculation for val/test
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if preprocess:
                X = preprocess(X)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss_log = f'Test loss ({batch+1}@{round(time.time()-start)}): {loss.item()}'
            print(loss_log)
            with open(LOG_FILE, 'a') as f:
                f.write(f'{loss_log}\n')
            avg_loss += loss.item()

    avg_loss /= len(dataloader)
    avg_loss_log = f'Test loss (avg@{round(time.time()-start)}): {avg_loss}'
    print(avg_loss_log)
    with open(LOG_FILE, 'a') as f:
        f.write(f'{avg_loss_log}\n')
