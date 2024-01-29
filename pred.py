import os
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.models import ResNet50_Weights

import config
from data import get_transform
from net import Net


def pred(img_file, n_model=None):
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)
    transform, _ = get_transform(preprocess)
    img = read_image(img_file, ImageReadMode.RGB)
    X = transform(img)
    X = X.view(1, *X.shape)

    model = Net().to(device, non_blocking=True)
    if not n_model:
        n_model = len(os.listdir(config.MODEL_DIR))
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, str(n_model), 'model.pth')))
    model.eval()
    with torch.no_grad():
        X = X.to(device, non_blocking=True)
        y = model(X)

    y = y.cpu().detach().numpy()[0]
    lat = y[0] * (config.MAX_LAT-config.MIN_LAT) + config.MIN_LAT
    lng = y[1] * (config.MAX_LNG-config.MIN_LNG) + config.MIN_LNG
    return lat, lng


if __name__ == '__main__':
    torch.manual_seed(42) # remove output randomness
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    lat, lng = pred('img.jpg', n_model=80)
    print(f'Pred: ({lat}, {lng})')
