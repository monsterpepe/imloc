# imloc
Neural network model trained to predict the coordinates (latitude, longitude) where an image is taken.
## Model
A ResNet50 model pretrained on ImageNet is used, and fine-tuned through training on a dataset of streetview images taken within a given latitude and longitude range. In total, the model has 40,297,538 trainable parameters. Training for ~128 epochs produces the best validation results.
![Train/test chart](/chart.png)
## Data
Streetview images and their coordinates are obtained from the [Mapillary API](https://www.mapillary.com/developer/api-documentation). In the current model, training is bound within the approximate coordinates of Singapore (see base_config.py for min/max of lat/lng). ~439,000 images used in total, 8:1:1 train:val:test split.
## To do
Current model performance is far from satisfactory.
1. Optimize neural net architecture - more/wider dense layers
2. Other pretrained models - EfficientNet, ViT etc.
3. Optimize training params - learning rate, betas, eps
4. Improve dataset quality (train/val/test) - low quality images jeopardize model performance
