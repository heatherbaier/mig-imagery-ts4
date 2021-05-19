import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import torch


class resnet50_mod(torch.nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x, ids):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = torch.cat((out, ids), dim = 1)
#         print(ids)
#         print(out)
#         print(torch.cat((out, ids), dim = 1))
#         print(.shape)
        out = self.fc(out)
        
        return out