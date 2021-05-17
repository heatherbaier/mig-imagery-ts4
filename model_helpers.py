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
import time


from image_loader import *



def loadImage(impath, root_dir):
    impath = os.path.join(root_dir, impath)
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB'))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig("./model_inputs2.png")
    plt.pause(.001)
    # plt.clf()

    
def plot_inputs(train, root_dir):
    a = 0
    ims = []
    for i in train:
        if a < 1:
            im_paths = list(i[0])
            images = [loadImage(i, root_dir) for i in im_paths]
            to_plot = torchvision.utils.make_grid(images)
            imshow(to_plot)
            plt.clf()
        else:
            break
        a += 1
        
        
        
def load_inputs(impath):
    impath = os.path.join("./model_imagery/", impath)
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB')).unsqueeze(0)
        
    
def mae(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return torch.abs(real - pred).mean()


def r2(true, pred):
    '''
    r2 = 1 - (RSS / TSS)
    R^2	=	coefficient of determination
    RSS	=	sum of squares of residuals
    TSS	=	total sum of squares
    '''
    m = torch.mean(true)
    TSS = sum((true - m) ** 2)
    RSS = sum((true - pred) ** 2)
    r2 = 1 - (RSS / TSS)
    return r2


def train_model(model, train, val, criterion, optimizer, epochs, batchSize, lr):

    start_time = time.perf_counter()

    best_mae = 9000000000000000000
    best_model_wts = deepcopy(model.state_dict())

    val_losses_plot = []


    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss, running_train_r2 = 0, 0, 0

                for inputs, output in train:
                    
                    print(c)

                    if len(inputs) == batchSize:
                        
                        inputs = [load_inputs(i) for i in list(inputs)]
                        inputs = torch.cat(inputs, dim = 0).to(f'cuda:{model.device_ids[0]}')#.to(device)
                        output = output.to(f'cuda:{model.device_ids[0]}')#.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = criterion(y_pred, output)  
                        
                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(outputs = loss, inputs = inputs, retain_graph = True)
                        loss.backward()
                        optimizer.step()
                        
                        running_train_mae += mae(y_pred, output).item()
                        running_train_loss += loss.item()
                        running_train_r2 += r2(output, y_pred).item()
                                                
                        # print(c)
                        c += 1

            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss, running_val_r2 = 0, 0, 0

                for inputs, output in val:

                    if len(inputs) == batchSize:

                        inputs = [load_inputs(i) for i in list(inputs)]
                        inputs = torch.cat(inputs, dim = 0).to(device)
                        output = output.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = criterion(y_pred, output)  

                        running_val_mae += mae(y_pred, output).item()
                        running_val_loss += loss.item()
                        running_val_r2 += r2(output, y_pred).item()
                        

                        # print(d)
                        d += 1
                        


                        
                        
        print("Epoch: ", epoch)  
        print("  Train:")
        print("    Loss: ", running_train_loss / c)      
        print("    MAE: ", running_train_mae / c)
        print("    R2: ", running_train_r2 / c)
        print("  Val:")
        print("    Loss: ", running_val_loss / d)      
        print("    MAE: ", running_val_mae / d)
        print("    R2: ", running_val_r2 / d)


        val_losses_plot.append(running_val_loss / d)
        

        if (running_val_mae / d) < best_mae:
            best_mae = (running_val_mae / d)
            best_model_wts = deepcopy(model.state_dict())

            print("  Saving current weights to epochs folder.")
        
        print("\n")

    end_time = time.perf_counter()
    print("Best MAE: ", best_mae)
    print("Training completed in: ", ((end_time - start_time) / 60) / 60, "hours.")
    print("\n")

    return best_model_wts, val_losses_plot




def train_model_onegpu(model, train, val, criterion, optimizer, epochs, batchSize, lr, device):

    start_time = time.perf_counter()

    best_mae = 9000000000000000000
    best_model_wts = deepcopy(model.state_dict())

    val_losses_plot = []


    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss, running_train_r2 = 0, 0, 0

                for inputs, output in train:
                    
                    print(c)

                    if len(inputs) == batchSize:
                        
                        inputs = [load_inputs(i) for i in list(inputs)]
                        inputs = torch.cat(inputs, dim = 0).to(device)#.to(device)
                        output = output.to(device)#.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = criterion(y_pred, output)  
                        
                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(outputs = loss, inputs = inputs, retain_graph = True)
                        loss.backward()
                        optimizer.step()
                        
                        running_train_mae += mae(y_pred, output).item()
                        running_train_loss += loss.item()
                        running_train_r2 += r2(output, y_pred).item()
                                                
                        # print(c)
                        c += 1

            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss, running_val_r2 = 0, 0, 0

                for inputs, output in val:

                    if len(inputs) == batchSize:

                        inputs = [load_inputs(i) for i in list(inputs)]
                        inputs = torch.cat(inputs, dim = 0).to(device)
                        output = output.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = criterion(y_pred, output)  

                        running_val_mae += mae(y_pred, output).item()
                        running_val_loss += loss.item()
                        running_val_r2 += r2(output, y_pred).item()
                        

                        # print(d)
                        d += 1
                        


                        
                        
        print("Epoch: ", epoch)  
        print("  Train:")
        print("    Loss: ", running_train_loss / c)      
        print("    MAE: ", running_train_mae / c)
        print("    R2: ", running_train_r2 / c)
        print("  Val:")
        print("    Loss: ", running_val_loss / d)      
        print("    MAE: ", running_val_mae / d)
        print("    R2: ", running_val_r2 / d)


        val_losses_plot.append(running_val_loss / d)
        

        if (running_val_mae / d) < best_mae:
            best_mae = (running_val_mae / d)
            best_model_wts = deepcopy(model.state_dict())

            print("  Saving current weights to epochs folder.")
        
        print("\n")

    end_time = time.perf_counter()
    print("Best MAE: ", best_mae)
    print("Training completed in: ", ((end_time - start_time) / 60) / 60, "hours.")
    print("\n")

    return best_model_wts, val_losses_plot