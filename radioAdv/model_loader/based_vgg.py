import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_loader.base_model import BaseModel


class Based_VGG(BaseModel):
    def __init__(self, output_dim):
        super(Based_VGG, self).__init__()

        # input(batch, 2, 128)
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels = 2, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv1(batch, 64, 64)
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv2(batch, 64, 32)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv3(batch, 64, 16)
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv4(batch, 64, 8)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = x.view(-1,2,128)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def loadBased_VGG(filepath):
    """[加载LeNet网络模型]

    Args:
        filepath ([str]): [LeNet的预训练模型所在的位置]

    Returns:
        [type]: [返回一个预训练的LeNet]
    """
    checkpoint = torch.load(filepath,map_location='cpu')
    model = Based_VGG(output_dim = 11)
    model.load_state_dict(checkpoint['state_dict'])  # 加载网络权重参数
    return model
    