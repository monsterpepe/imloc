import matplotlib.pyplot as plt
import numpy as np
import os
import config

if __name__ == '__main__':
    model_dir = 'model'
    # model_dir = 'old/resnet50-4-lr0.003'
    n_models = len(os.listdir(model_dir))
    train_losses = []
    val_losses = []
    avg_val_losses = []
    models = []
    for model in range(1, n_models+1):
        try:
            with open(os.path.join(model_dir, str(model), 'log.txt')) as f:
                s = f.read()

            log = s.split('\n')
            for n, i in enumerate(log):
                if 'Train loss' in i:
                    if n % 10:
                        continue
                    loss = float(i.split(': ')[-1])
                    train_losses.append(loss)
                elif 'Test loss' in i:
                    if 'avg' in i:
                        loss = float(i.split(': ')[-1])
                        avg_val_losses.append(loss)
                        models.append(model)
                    else:
                        if n % 10: # too much data
                            continue
                        loss = float(i.split(': ')[-1])
                        val_losses.append(loss)
        except FileNotFoundError:
            print(f'No log for model {model}')

    win = 500
    train_sma = []
    for i in range(len(train_losses)-win+1):
        train_sma.append(np.mean(train_losses[i:i+win]))

    win = 100
    val_sma = []
    for i in range(len(val_losses)-win+1):
        val_sma.append(np.mean(val_losses[i:i+win]))

    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(ncols=2, nrows=2)
    fig.delaxes(_)
    ax1.plot(range(len(train_sma)), train_sma)
    ax1.title.set_text('Train loss')
    ax2.plot(range(len(val_sma)), val_sma)
    ax2.title.set_text('Val loss')
    ax3.plot(models, avg_val_losses)
    ax3.title.set_text('Avg val loss per epoch')
    plt.show()
