from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import json
import os


class MigrationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mig_json, root_dir, ref_file):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "SEP", "OCT", "NOV", "DEC"]
        self.root_dir = root_dir
        self.image_names = os.listdir(root_dir)#[0:7000]
        self.image_paths, self.migrants, self.ref_encodes = [], [], []
        self.muni_ids = []

            
        m = open(mig_json,)
        self.mig_data = json.load(m)
        
        r = open(ref_file,)
        self.ref_data = json.load(r)

        for month in self.months:
            print(month)
            month_images = [im for im in self.image_names if im.endswith(month + ".png")]
            munis = [i.split("_")[0] for i in month_images]
            migs = [self.mig_data[i] if i in self.mig_data.keys() else 0 for i in munis]
            ref_encodes = [self.ref_data[i] if i in self.ref_data.keys() else [1] * 2269 for i in munis]
            # weights = [1 / self.weight_data[i] if i in self.weight_data.keys() else 0 for i in munis]
            [self.image_paths.append(i) for i in month_images]
            [self.migrants.append(i) for i in migs]
            [self.ref_encodes.append(i) for i in ref_encodes]
            [self.muni_ids.append(int(i.split("-")[3].strip("B"))) for i in munis]

        self.data = [(self.image_paths[i], self.migrants[i]) for i in range(0, len(self.image_paths))]


    def loadImage(self, impath):
        impath = os.path.join(self.root_dir, impath)
        to_tens = transforms.ToTensor()
        return to_tens(Image.open(impath).convert('RGB'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        return self.loadImage(path), self.labels[index]
    
    
    
def train_test_split(X, y, w, r, split):

    train_num = int(len(X) * split)
    val_num = int(len(X) - train_num)

    all_indices = list(range(0, len(X)))
    train_indices = random.sample(range(len(X)), train_num)
    val_indices = list(np.setdiff1d(all_indices, train_indices))

    x_train, x_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    w_train, w_val = w[train_indices], w[val_indices]
    ref_train, ref_val = r[train_indices], r[val_indices]

    return x_train, y_train, x_val, y_val, w_train, w_val, ref_train, ref_val
    
    
    
# d = MigrationDataset("./data/migration_data.json", "./model_imagery", "./image_weights.json")

# x_train, y_train, x_val, y_val = train_test_split(np.array(d.image_paths), np.array(d.migrants), .75)

# print(len(x_train))
# print(len(y_train))
# print(len(x_val))
# print(len(y_val))

# train = [(k,v) for k,v in zip(x_train, y_train)]
# val = [(k,v) for k,v in zip(x_val, y_val)]

# print(len(train))
# print(len(val))

# batchSize = 32

# train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
# val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)

# print('done')
