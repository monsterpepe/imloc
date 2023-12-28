import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50.fc = nn.Identity()
        self.regression_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048), # Test remove one linear layer
            nn.ReLU(),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = self.resnet50(x)
        return self.regression_layer(x)


if __name__ == '__main__':
    net = ResNet50Pretrained()
    print(net)
    with open('net.txt', 'w') as f:
        f.write(net.__repr__())