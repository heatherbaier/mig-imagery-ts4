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
    impath = os.path.join("./model_imagery2/", impath)
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB')).unsqueeze(0)
        
    
def mae(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return torch.abs(real - pred).mean()




def muni_mae(pred, true, ref_ids, model, device):
#     batch_mae = torch.tensor([0]).to(device)
    unique_ids = torch.unique(ref_ids)
#     print("REF IDS: ", ref_ids)
#     print("UNIQUE REF IDS: ", unique_ids)
    for i in unique_ids:
        indices = torch.nonzero(ref_ids == i, as_tuple = True)[0]
        true_sum = torch.mean(torch.index_select(true, 0, indices))
        pred_sum = torch.mean(torch.index_select(pred, 0, indices))
#         print("TRUE: ", true_sum)
#         print("PRED: ", pred_sum)
        true_m_pred = torch.abs(true_sum - pred_sum).unsqueeze(0)
#         print(true_m_pred)
        try:
#             print('try')
            batch_mae = torch.cat((batch_mae, true_m_pred))
        except:
#             print("except")
            batch_mae = true_m_pred
#         print(batch_mae)
    batch_mae = torch.mean(batch_mae)
    return batch_mae
    
    
    
    
def custom_loss(pred, true, ref_ids, smodel, device):
#     loss = torch.tensor([0]).to(device)
    unique_ids = torch.unique(ref_ids)
    for i in unique_ids:
        indices = torch.nonzero(ref_ids == i, as_tuple = True)[0]
        true_sum = torch.sum(torch.index_select(true, 0, indices))
        pred_sum = torch.sum(torch.index_select(pred, 0, indices))
        true_m_pred = torch.abs(true_sum - pred_sum).unsqueeze(0)
        try:
            loss = torch.cat((loss, true_m_pred))
        except:
            loss = true_m_pred
    loss = torch.sum(loss)
    return loss
    
    

    
def train_model(model, train, val, criterion, optimizer, epochs, batchSize, lr, device):

    start_time = time.perf_counter()

    best_mae = 9000000000000000000
    best_model_wts = deepcopy(model.state_dict())

    val_losses_plot = []

    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss, running_train_r2 = 0, 0, 0

                for inputs, output, encoded_ids, ref_ids, coords in train:
                                                            
                    try:

                        # Load inputs
                        inputs = torch.cat([load_inputs(i) for i in list(inputs)], dim = 0)

                        # Format everything as tensors with correct shape
                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.tensor(output, dtype = torch.float32, requires_grad = True).view(-1, 1)
                        encoded_ids = torch.tensor(encoded_ids, dtype = torch.float32, requires_grad = True)
                        ref_ids = torch.tensor(ref_ids, dtype = torch.long).view(-1, 1).to(device)

                        # Send everything to devices
                        inputs = inputs.to(device)
                        output = output.to(device)
                        coords = coords.to(device)
                        encoded_ids = encoded_ids.to(device)

                        # Forward pass
                        y_pred = model(inputs, encoded_ids, coords)
                        loss = custom_loss(y_pred, output, ref_ids, model, device)

                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Update all the stats
        #                     print(muni_mae(y_pred, output, ref_ids, model, device).item())
                        running_train_mae += muni_mae(y_pred, output, ref_ids, model, device).item()
                        running_train_loss += loss.item()

                        c += 1
                    
                    
                    
#                     print("\n")
                    
                        
                            
                    except Exception as e:

                        print("Bad data: ", e)
                        
                        
            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss, running_val_r2 = 0, 0, 0

                for inputs, output, encoded_ids, ref_ids, coords in val:
                    
                    try:

                        # Load inputs
                        inputs = torch.cat([load_inputs(i) for i in list(inputs)], dim = 0)

                        # Format everything as tensors with correct shape
                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.tensor(output, dtype = torch.float32, requires_grad = True).view(-1, 1)
                        encoded_ids = torch.tensor(encoded_ids, dtype = torch.float32, requires_grad = True)
                        ref_ids = torch.tensor(ref_ids, dtype = torch.long).view(-1, 1).to(device)

                        # Send everything to devices
                        inputs = inputs.to(device)
                        output = output.to(device)
                        coords = coords.to(device)
                        encoded_ids = encoded_ids.to(device)

                        # Forward pass
                        y_pred = model(inputs, encoded_ids, coords)
                        loss = custom_loss(y_pred, output, ref_ids, model, device)

                        running_val_mae += mae(y_pred, output).item()
                        running_val_loss += loss.item()

                        d += 1

                    except Exception as e:
                        
                        print("Bad data: ", e)

                        
                        
        print("Epoch: ", epoch)  
        print("  Train:")
        print("    Loss: ", running_train_loss / c)      
        print("    MAE: ", running_train_mae / c)
        print("  Val:")
        print("    Loss: ", running_val_loss / d)      
        print("    MAE: ", running_val_mae / d)


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





def train_model_nocoords(model, train, val, criterion, optimizer, epochs, batchSize, lr, device):

    start_time = time.perf_counter()

    best_mae = 9000000000000000000
    best_model_wts = deepcopy(model.state_dict())

    val_losses_plot = []

    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss, running_train_r2 = 0, 0, 0

                for inputs, output, encoded_ids, ref_ids, coords in train:
                                                            
                    try:

                        # Load inputs
                        inputs = torch.cat([load_inputs(i) for i in list(inputs)], dim = 0)

                        # Format everything as tensors with correct shape
                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.tensor(output, dtype = torch.float32, requires_grad = True).view(-1, 1)
                        encoded_ids = torch.tensor(encoded_ids, dtype = torch.float32, requires_grad = True)
                        ref_ids = torch.tensor(ref_ids, dtype = torch.long).view(-1, 1).to(device)

                        # Send everything to devices
                        inputs = inputs.to(device)
                        output = output.to(device)
                        coords = coords.to(device)
                        encoded_ids = encoded_ids.to(device)

                        # Forward pass
                        y_pred = model(inputs, encoded_ids)
                        loss = custom_loss(y_pred, output, ref_ids, model, device)

                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Update all the stats
        #                     print(muni_mae(y_pred, output, ref_ids, model, device).item())
                        running_train_mae += muni_mae(y_pred, output, ref_ids, model, device).item()
                        running_train_loss += loss.item()

                        c += 1
                    
                    
                    
#                     print("\n")
                    
                        
                            
                    except Exception as e:

                        print("Bad data: ", e)
                        
                        
            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss, running_val_r2 = 0, 0, 0

                for inputs, output, encoded_ids, ref_ids, coords in val:
                    
                    try:

                        # Load inputs
                        inputs = torch.cat([load_inputs(i) for i in list(inputs)], dim = 0)

                        # Format everything as tensors with correct shape
                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.tensor(output, dtype = torch.float32, requires_grad = True).view(-1, 1)
                        encoded_ids = torch.tensor(encoded_ids, dtype = torch.float32, requires_grad = True)
                        ref_ids = torch.tensor(ref_ids, dtype = torch.long).view(-1, 1).to(device)

                        # Send everything to devices
                        inputs = inputs.to(device)
                        output = output.to(device)
                        coords = coords.to(device)
                        encoded_ids = encoded_ids.to(device)

                        # Forward pass
                        y_pred = model(inputs, encoded_ids)
                        loss = custom_loss(y_pred, output, ref_ids, model, device)

                        running_val_mae += mae(y_pred, output).item()
                        running_val_loss += loss.item()

                        d += 1

                    except Exception as e:
                        
                        print("Bad data: ", e)

                        
                        
        print("Epoch: ", epoch)  
        print("  Train:")
        print("    Loss: ", running_train_loss / c)      
        print("    MAE: ", running_train_mae / c)
        print("  Val:")
        print("    Loss: ", running_val_loss / d)      
        print("    MAE: ", running_val_mae / d)


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




def eval(data, model, device):
    preds = []
    trues = []
    im_paths = []
    ids = []

    for obs in data:
                
#         try:

        muni_id = obs[3].item()
        cur_path = obs[0][0]
        cur_true = obs[1].item()
        cur_encode = torch.tensor(obs[2], dtype = torch.float32, requires_grad = True)
        input = load_inputs(cur_path)
        coords = obs[4]
        pred = model(input.to(device), cur_encode.to(device), coords.to(device)).item()

        preds.append(pred)
        trues.append(cur_true)
        im_paths.append(cur_path)
        ids.append(muni_id)
            
#         except Exception as e:
#             print("Bad data: ", e)
    
    
    preds_df = pd.DataFrame()
    preds_df['muni'] = ids
    preds_df['im_path'] = im_paths
    preds_df['true'] = trues   
    preds_df['pred'] = preds
    
    return preds_df



def r2_np(true, pred):
    '''
    r2 = 1 - (RSS / TSS)
    R^2	=	coefficient of determination
    RSS	=	sum of squares of residuals
    TSS	=	total sum of squares
    '''
    m = np.mean(true)
    TSS = sum((true - m) ** 2)
    RSS = sum((true - pred) ** 2)
    r2 = 1 - (RSS / TSS)
    return r2


def mae_np(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return abs(real - pred).mean()