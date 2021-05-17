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
BATCH_SIZE = 16


d = MigrationDataset(MIG_JSON, ROOT_DIR, WEIGHTS_JSON)

x_train, y_train, x_val, y_val = train_test_split(np.array(d.image_paths), np.array(d.migrants), .75)

train = [(k,v) for k,v in zip(x_train, y_train)]
val = [(k,v) for k,v in zip(x_val, y_val)]



train_dl = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)

print(len(train))
print(len(val))

print("Done loading imagery.")

plot_inputs(train_dl, ROOT_DIR)




lr = 1e-4
epochs = 1
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda", [1,2])
# print(device)
model = models.resnet50(pretrained=True)#.to(device)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)
# model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[1,2,3])
model = model.to(f'cuda:{model.device_ids[0]}')
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


model_wts, val_losses_plot = train_model(model, train_dl, val_dl, criterion, optimizer, epochs, BATCH_SIZE, lr)
