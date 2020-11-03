import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel


class Baseline_CNN(BaseModel):
    def __init__(self, output_dim):
        super(Baseline_CNN, self).__init__()
        # input(batch, 1, 2, 128)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels= 1, out_channels = 64, kernel_size= (1, 8), stride= 1, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (1,2))
        )
        # after conv1(batch, 128, 2, 60)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels= 64, out_channels= 32, kernel_size=(2,8), stride= 2, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (1,2))
        )
        # afer conv2(batch, 64, 1, 13)
        # self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 32*13, out_features= 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 64, out_features= output_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        # print('conv1', x.shape)
        x = self.conv2(x)
        # print('conv2', x.shape)
        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x