import numpy as np

def num_params(model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in params])

if __name__ == '__main__':
    from net import Net
    model = Net()
    print(num_params(model))
