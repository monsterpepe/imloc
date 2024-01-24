import numpy as np
from net import Net

def num_params(model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in params])

model = Net()
print(num_params(model))