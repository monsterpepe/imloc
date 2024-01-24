import os
import time
import torch
from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights

import config
from data import get_loaders

# 1. More dense layers
# 2. Wider dense layers?
# 3. Other pretrained models ie EfficientNet, Inception

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.fc = nn.Identity()
        self.regression_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048), # Test remove one linear layer
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
        optimizer.zero_grad(set_to_none=True) # ~2s time reduction
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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)
    train_loader, val_loader, test_loader = get_loaders(preprocess)

    model = Net().to(device, non_blocking=True)
    print(model)
    n_model = len(os.listdir(config.MODEL_DIR))
    os.mkdir(os.path.join(config.MODEL_DIR, str(n_model+1)))
    if n_model: # continue training from most old model
        model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, str(n_model), 'model.pth')))
    log_file = os.path.join(config.MODEL_DIR, str(n_model+1), 'log.txt')
    mse = nn.MSELoss()
    adam = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    start = time.time()
    train(train_loader, model, mse, adam, device=device, log_file=log_file)
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f'model.pth'))
    print('Saved model')
    test(val_loader, model, mse, device=device, log_file=log_file)
