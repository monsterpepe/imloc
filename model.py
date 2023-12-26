import os
import time

import torch
from torch import nn, optim

from data import make_dataloaders
from net import ResNet50Pretrained, ResNet50_Weights, resnet50
import config

start = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

n_model = 1
for i in os.listdir(config.MODEL_DIR):
    if i.endswith('.pth'):
        n_model += 1

LOG_FILE = os.path.join(config.MODEL_DIR, f'log_{n_model}.txt')


def train(dataloader, model, loss_fn, optimizer, preprocess=None):
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # update weights

        loss_log = f'Train loss ({batch+1}@{round(time.time()-start)}): {loss.item()}'
        print(loss_log)
        with open(LOG_FILE, 'a') as f:
            f.write(f'{loss_log}\n')


def test(dataloader, model, loss_fn, preprocess=None):
    print('Testing...')
    model.eval()
    avg_loss = 0
    with torch.no_grad():
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


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = make_dataloaders()
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)

    model = ResNet50Pretrained().to(device, non_blocking=True)
    mse = nn.MSELoss()
    adam = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    for n_epoch in range(config.EPOCHS):
        epoch_log = f'Epoch: {n_epoch+1}'
        print(epoch_log)
        with open(LOG_FILE, 'a') as f:
            f.write(f'{epoch_log}\n')

        train(train_dataloader, model, mse, adam, preprocess=preprocess)

        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f'model_{n_model}_{n_epoch+1}.pth'))
        print(f'Saved model ({round(time.time()-start)})')

        test(val_dataloader, model, mse, preprocess=preprocess)
