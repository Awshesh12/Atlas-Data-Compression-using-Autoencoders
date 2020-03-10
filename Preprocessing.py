# Taken from Eric Wulff's code
# Preprocessing before the creation of Neural net

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

force_cpu = False

if force_cpu:
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
print('Using device:', device)

import sys

train = pd.read_pickle('all_jets_train_4D_100_percent.pkl')
test = pd.read_pickle('all_jets_test_4D_100_percent.pkl')

train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

train_x = train
test_x = test
train_y = train_x
test_y = test_x
train_x = train_x
test_x = test
train_y = train_x
test_y = test_x

train_ds = TensorDataset(torch.tensor(train_x.values,dtype=torch.float), torch.tensor(train_y.values,dtype=torch.float))
valid_ds = TensorDataset(torch.tensor(test_x.values,dtype=torch.float), torch.tensor(test_y.values,dtype=torch.float))

def fit(epochs, model, loss_func, opt, train_dl, valid_dl,device):
    since = time.time()
    epochs_train_loss = []
    epochs_val_loss = []
    for epoch in range(epochs):
        running_train_loss = 0.
        epoch_start = time.perf_counter()
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss, lenxb = loss_batch(model, loss_func, xb, yb, opt)
            running_train_loss += np.multiply(loss, lenxb)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb_tmp.to(device), yb_tmp.to(device)) for xb_tmp, yb_tmp in valid_dl]
            )
        train_loss = running_train_loss / len(train_dl.dataset)
        epochs_train_loss.append(train_loss)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        epochs_val_loss.append(val_loss)
        if(epoch % 1 == 0):
            current_time = time.perf_counter()
            delta_t = current_time - epoch_start
            # print('Epoch ' + str(epoch) + ':', 'Validation loss = ' + str(val_loss) + ' Time: ' + str(datetime.timedelta(seconds=delta_t)))
            print('Epoch: {:d} Train Loss: {:.3e} Val Loss: {:.3e}, Time: {}'.format(epoch, train_loss, val_loss, str(datetime.timedelta(seconds=delta_t))))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    a=pd.DataFrame({'Epoch': np.arange(epochs), 'train_loss': np.array(epochs_train_loss), 'val_loss': np.array(epochs_val_loss), 'epoch_time': delta_t})
    return (a)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
