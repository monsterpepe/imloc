import os
import time
import torch
from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights

import config
from data import get_loaders


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.fc = nn.Identity()
        self.regression_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = self.resnet50(x)
        return self.regression_layer(x)


def train(dataloader, model, loss_fn, optimizer, device='cuda', log_file=None):
    print('Training...')
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Compute prediction error (forward pass)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backpropagation (backward pass)
        optimizer.zero_grad(set_to_none=True) # set_to_none for time reduction
        loss.backward()
        optimizer.step() # update weights

        loss_log = f'Train loss ({batch+1}@{round(time.time()-start)}): {loss.item()}'
        print(loss_log)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f'{loss_log}\n')


def test(dataloader, model, loss_fn, device='cuda', log_file=None):
    print('Testing...')
    model.eval()
    avg_loss = 0
    with torch.no_grad(): # disable gradient calculation for val/test
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss_log = f'Test loss ({batch+1}@{round(time.time()-start)}): {loss.item()}'
            print(loss_log)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f'{loss_log}\n')
            avg_loss += loss.item()

    avg_loss /= len(dataloader)
    avg_loss_log = f'Test loss (avg@{round(time.time()-start)}): {avg_loss}'
    print(avg_loss_log)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'{avg_loss_log}\n')


euclidean = lambda a, b: torch.sqrt(((a-b)**2).sum(1)).mean()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if config.MODEL_DIR not in os.listdir():
        os.mkdir(config.MODEL_DIR)

    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)
    train_loader, val_loader, _ = get_loaders(preprocess)
    model = Net().to(device, non_blocking=True)

    while True:
        print(model)
        n_model = len(os.listdir(config.MODEL_DIR))
        if n_model: # continue training from last model
            model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, str(n_model), 'model.pth')))
        # mse = nn.MSELoss()
        adam = optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        os.mkdir(os.path.join(config.MODEL_DIR, str(n_model+1)))
        log_file = os.path.join(config.MODEL_DIR, str(n_model+1), 'log.txt')

        print(f'Epoch: {n_model+1}')
        start = time.time()
        train(train_loader, model, euclidean, adam, device=device, log_file=log_file)
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, str(n_model+1), f'model.pth'))
        print('Saved model')
        test(val_loader, model, euclidean, device=device, log_file=log_file)
