import matplotlib.pyplot as plt
import numpy as np
import os
import config

models = os.listdir(config.MODEL_DIR)
train_losses = []
val_losses = []
avg_val_losses = []
for model in models:
    with open(os.path.join(config.MODEL_DIR, str(model), 'log.txt')) as f:
        s = f.read()

    log = s.split('\n')
    for i in log:
        if 'Train loss' in i:
            loss = float(i.split(': ')[-1])
            train_losses.append(loss ** 0.5)

        elif 'Test loss' in i:
            if 'avg' in i:
                loss = float(i.split(': ')[-1])
                avg_val_losses.append(loss ** 0.5)
            else:
                loss = float(i.split(': ')[-1])
                val_losses.append(loss ** 0.5)

win = 100
train_sma = []
for i in range(len(train_losses)-win+1):
    train_sma.append(np.mean(train_losses[i:i+win]))

win = 20
val_sma = []
for i in range(len(val_losses)-win+1):
    val_sma.append(np.mean(val_losses[i:i+win]))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(range(len(train_sma)), train_sma)
ax2.plot(range(len(val_sma)), val_sma)
ax3.plot(range(1, len(avg_val_losses)+1), avg_val_losses)
plt.show()