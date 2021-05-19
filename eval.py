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
from resnet50_mod import *



ROOT_DIR = "./model_imagery"
MIG_JSON = "./data/migration_data.json"
WEIGHTS_JSON = "./data/encoded_munis.json"
BATCH_SIZE = 16


d = MigrationDataset(MIG_JSON, ROOT_DIR, WEIGHTS_JSON)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained = True)#.to(device)
model = resnet50_mod(resnet50)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs + 2269, 1)
model = model.to(device)


checkpoint = torch.load("./epochs/model_wl_v2_epoch10_alldata.torch")
all_params = [(k,v) for k,v in checkpoint['model_state_dict'].items()]
all_params = dict([(".".join(i[0].split(".")[1:]), i[1]) for i in all_params])

model.load_state_dict(all_params)


preds = []
trues = []
im_paths = []

for obs in range(0, len(d.image_paths)):
    try:
        cur_path = d.image_paths[obs]
        cur_true = d.migrants[obs]
        cur_encode = d.ref_encodes[obs]
        cur_encode = torch.reshape(torch.tensor(cur_encode, dtype = torch.float32, requires_grad = True), (1, 2269))
        input = load_inputs(cur_path)
        pred = model(input.to(device), cur_encode.to(device)).item()
        print(pred)

        preds.append(pred)
        trues.append(cur_true)
        im_paths.append(cur_path)
        
    except Exception as e:
        print("Bad.", e)
    
    
preds_df = pd.DataFrame()
preds_df['impath'] = im_paths
preds_df['true'] = trues   
preds_df['pred'] = preds
preds_df.to_csv("./preds_df2.csv", index = False)