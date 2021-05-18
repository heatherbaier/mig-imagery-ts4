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


ROOT_DIR = "./model_imagery"
MIG_JSON = "./data/migration_data.json"
WEIGHTS_JSON = "./image_weights.json"
BATCH_SIZE = 256


d = MigrationDataset(MIG_JSON, ROOT_DIR, WEIGHTS_JSON)

x_train, y_train, x_val, y_val, w_train, w_val = train_test_split(np.array(d.image_paths), np.array(d.migrants), np.array(d.weights), .75)

train = [(k,v,w) for k,v,w in zip(x_train, y_train, w_train)]
val = [(k,v,w) for k,v,w in zip(x_val, y_val, w_val)]

train_dl = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)

print(len(train_dl))
print(len(val_dl))


print("Done loading imagery.")


plot_inputs(train_dl, ROOT_DIR)


lr = 1e-4
epochs = 25
model = models.resnet50(pretrained = True)#.to(device)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)
model = torch.nn.DataParallel(model, device_ids = [0,1,2,3])
model = model.to(f'cuda:{model.device_ids[0]}')
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


model_wts, val_losses_plot = train_model(model, train_dl, val_dl, criterion, optimizer, epochs, BATCH_SIZE, lr)


model.load_state_dict(model_wts)


train_dl = torch.utils.data.DataLoader(train, batch_size = 1, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = 1, shuffle = True)




print("Evaluating.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eval(val_dl, model, device).to_csv("./val_preds.csv", index = False)
eval(train_dl, model, device).to_csv("./train_preds.csv", index = False)


model.load_state_dict(model_wts)

torch.save({
            'epoch': 25,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./trained_models/model_nl_v1_alldata.torch")
