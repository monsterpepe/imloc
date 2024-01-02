import os
import torch
from torch import nn, optim

from data import make_dataloaders
from train import train, validate
from net import ResNet50Pretrained, ResNet50_Weights
import config

# Train one epoch
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_dataloader, val_dataloader, test_dataloader = make_dataloaders()
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)

    model = ResNet50Pretrained().to(device, non_blocking=True)

    n_model = len(os.listdir(config.MODEL_DIR))
    os.mkdir(os.path.join(config.MODEL_DIR, str(n_model+1)))
    if n_model: # continue training from most recent model
        model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, str(n_model), 'model.pth')))

    mse = nn.MSELoss()
    adam = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    train(train_dataloader, model, mse, adam, preprocess=preprocess, device=device)

    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, str(n_model+1), f'model.pth'))
    print(f'Saved model')

    validate(val_dataloader, model, mse, preprocess=preprocess, device=device)
