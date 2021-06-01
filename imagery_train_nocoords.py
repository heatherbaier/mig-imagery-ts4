import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import torchvision.models as models
# from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import torchvision
import importlib
import argparse
# import sklearn
import random
import torch
import math
import json

from image_loader import *
from model_helpers import *
from resnet50_mod_nocoords import *


ROOT_DIR = "./model_imagery2"
MIG_JSON = "./data/migration_data.json"
WEIGHTS_JSON = "./data/encoded_munis.json"
BATCH_SIZE = 256


d = MigrationDataset(MIG_JSON, ROOT_DIR, WEIGHTS_JSON)



x_train, y_train, x_val, y_val, id_train, id_val, ref_train, ref_val, c_train, c_val = train_test_split(np.array(d.image_paths), np.array(d.migrants), np.array(d.ref_encodes), np.array(d.muni_ids), np.array(d.coords, dtype = np.float32), .75)

train = [(k,v,i,r,c) for k,v,i,r,c in zip(x_train, y_train, id_train, ref_train, c_train)]
val = [(k,v,i,r,c) for k,v,i,r,c in zip(x_val, y_val, id_val, ref_val, c_val)]

train_dl = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)

print(len(train_dl))
print(len(val_dl))

print("Done loading imagery.")


plot_inputs(train_dl, ROOT_DIR)


lr = 1e-3
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(pretrained = True)#.to(device)
num_ftrs = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(num_ftrs + 2331, 1)
model = torch.nn.DataParallel(resnet50_mod_nocoords(resnet50)).to(device)
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


model_wts, val_losses_plot = train_model_nocoords(model, train_dl, val_dl, criterion, optimizer, epochs, BATCH_SIZE, lr, device)


model.load_state_dict(model_wts)


torch.save({
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./trained_models/r50_nocoords_10_epochs.torch")



train_dl = torch.utils.data.DataLoader(train, batch_size = 1, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = 1, shuffle = True)


model.load_state_dict(model_wts)


print("Evaluating.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_preds = eval(val_dl, model, device)
train_preds = eval(train_dl, model, device)


val_preds.to_csv("./val_preds_r50_nocoords_10_epochs.csv", index = False)
train_preds.to_csv("./train_preds_r50_nocoords_10_epochs.csv", index = False)



avg_val_preds = pd.DataFrame(val_preds.groupby(val_preds['muni'])['true', 'pred'].mean()).reset_index()
print("VAL R2:  ", r2_np(avg_val_preds['true'], avg_val_preds['pred']))
print("VAL MAE: ", mae_np(avg_val_preds['true'], avg_val_preds['pred']))
plt.scatter(avg_val_preds['true'], avg_val_preds['pred'])
plt.title("Validation")
plt.savefig("./avg_val_preds_r50_nocoords_10_epochs.png")
plt.clf()


avg_train_preds = pd.DataFrame(train_preds.groupby(train_preds['muni'])['true', 'pred'].mean()).reset_index()
print("TRAIN R2:  ", r2_np(avg_train_preds['true'], avg_train_preds['pred']))
print("TRAIN MAE: ", mae_np(avg_train_preds['true'], avg_train_preds['pred']))
plt.scatter(avg_train_preds['true'], avg_train_preds['pred'])
plt.title("Training")
plt.savefig("./avg_train_preds_r50_nocoords_10_epochs.png")
plt.clf()
