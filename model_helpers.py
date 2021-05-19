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


# def weighted_loss(pred, true, weight):
#     return torch.sum((abs(pred - true) * weight))

#     lossList = []
#     for municipality in pred:
#         lossList.append(torch.sum((abs(pred[municipality] - true[municipality]))))

#     loss = sum(lossList)
    
    
    
def custom_loss(pred, true, ref_ids, model):
    loss = torch.tensor([0]).to(f'cuda:{model.device_ids[0]}')#.to(device)
    unique_ids, inverse_indices = torch.unique(torch.tensor(ref_ids, dtype = torch.long), return_inverse=True)
    for i in unique_ids:
        indices = (ref_ids == i).nonzero(as_tuple = True)[0]
        true_vals = torch.squeeze(torch.index_select(true, 0, indices), 1)[0]
        pred_vals = torch.mean(torch.index_select(pred, 0, indices))
        cur_diff = pred_vals - true_vals.unsqueeze(0)
        loss = torch.cat((loss, cur_diff))
        
    loss = torch.abs(torch.sum(loss))
    
    return loss
    
    

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

                for inputs, output, encoded_ids, ref_ids in train:
                                        
                    try:

                        if len(inputs) == batchSize:

                            inputs = [load_inputs(i) for i in list(inputs)]
                            inputs = torch.cat(inputs, dim = 0).to(f'cuda:{model.device_ids[0]}')#.to(device)
                            output = output.to(f'cuda:{model.device_ids[0]}')#.to(device)
                            encoded_ids = encoded_ids.to(f'cuda:{model.device_ids[0]}')#.to(device)

                            inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                            output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))
                            encoded_ids = torch.reshape(torch.tensor(encoded_ids, dtype = torch.float32, requires_grad = True), (batchSize, 2269))
                            ref_ids = torch.reshape(torch.tensor(ref_ids, dtype = torch.float32, requires_grad = True), (batchSize, 1)).to(f'cuda:{model.device_ids[0]}')

                            # Forward pass
                            y_pred = model(inputs, encoded_ids)
                            loss = custom_loss(y_pred, output, ref_ids, model)

                            # Zero gradients, perform a backward pass, and update the weights.
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Update all the stats
                            running_train_mae += mae(y_pred, output).item()
                            running_train_loss += loss.item()
                            running_train_r2 += r2(output, y_pred).item()

                            c += 1
                        
                            
                    except Exception as e:

                        print("Bad data: ", e)
                        
#                 adjlkasjl

            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss, running_val_r2 = 0, 0, 0

                for inputs, output, encoded_ids, ref_ids in val:
                    
                    try:

                        if len(inputs) == batchSize:

                            inputs = [load_inputs(i) for i in list(inputs)]
                            inputs = torch.cat(inputs, dim = 0).to(f'cuda:{model.device_ids[0]}')#.to(device)
                            output = output.to(f'cuda:{model.device_ids[0]}')#.to(device)
                            encoded_ids = encoded_ids.to(f'cuda:{model.device_ids[0]}')#.to(device)

                            inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                            output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))
                            encoded_ids = torch.reshape(torch.tensor(encoded_ids, dtype = torch.float32, requires_grad = True), (batchSize, 2269))
                            ref_ids = torch.reshape(torch.tensor(ref_ids, dtype = torch.float32, requires_grad = True), (batchSize, 1)).to(f'cuda:{model.device_ids[0]}')


                            # Forward pass
                            y_pred = model(inputs, encoded_ids)
                            loss = custom_loss(y_pred, output, ref_ids, model)

                            running_val_mae += mae(y_pred, output).item()
                            running_val_loss += loss.item()
                            running_val_r2 += r2(output, y_pred).item()

                            d += 1
                        
                    except Exception as e:
                        
                        print("Bad data: ", e)

                        
                        
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
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                    }, "./epochs/model_wl_v4_epoch" + str(epoch) + "_alldata.torch")

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

    for obs in data:
        try:
            cur_path = obs[0][0]
            cur_true = obs[1]
            cur_encode = obs[2]
            cur_encode = torch.reshape(torch.tensor(cur_encode, dtype = torch.float32, requires_grad = True), (1, 2269))
            input = load_inputs(cur_path)
            pred = model(input.to(device), cur_encode.to(device)).item()
            preds.append(pred)
            trues.append(cur_true.item())
            im_paths.append(cur_path)
        except Exception as e:
            print(e)
    
    
    preds_df = pd.DataFrame()
    preds_df['impath'] = im_paths
    preds_df['true'] = trues   
    preds_df['pred'] = preds
    
    return preds_df
