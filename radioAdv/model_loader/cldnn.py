import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_loader.base_model import BaseModel

class CLDNN(BaseModel):
    def __init__(self, output_dim):
        super(CLDNN, self).__init__()
        # input(batch, 1, 2, 128)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels= 1, out_channels = 256, kernel_size= (1, 3), stride= 1, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (1,2))
        )
        # after conv1(batch, 256, 2, 63)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels= 256, out_channels= 80, kernel_size=(2,3), stride= 1, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (1,2))
        )
        # afer conv2(batch, 256, 1, 30)
        # self.conv3 = nn.Sequential(
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(in_channels= 256, out_channels= 80, kernel_size=(1,3), stride= 1, padding= 0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size= (1,2))
        # )
        # # afer conv3(batch, 80, 1, 14)
        # self.conv4 = nn.Sequential(
        #     nn.BatchNorm2d(80),
        #     nn.Conv2d(in_channels= 80, out_channels= 80, kernel_size=(1,3), stride= 1, padding= 0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size= (1,2))
        # )
        # afer conv4(batch, 80, 1, 6)
        # reshape(batch, 80, 6)
        self.lstm1 = nn.Sequential(
            nn.GRU(input_size= 30, hidden_size= 50, num_layers= 3, batch_first= True)
        )
        self.tanh = nn.Tanh()
        # after lstm1 (batch, 80, 50)
        # 取最后一个状态的feature (batch, 50)
        # self.dropout = nn.Dropout2d(0.5)
        # after reshape (batch, 80*50)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 80*50, out_features= 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= output_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(x.shape[0], 80, 30)
        x, _ = self.lstm1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.tanh(x)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def loadCLDNN(filepath):
    """[加载LeNet网络模型]

    Args:
        filepath ([str]): [LeNet的预训练模型所在的位置]

    Returns:
        [type]: [返回一个预训练的LeNet]
    """
    checkpoint = torch.load(filepath)
    model = CLDNN(output_dim = 11)
    model.load_state_dict(checkpoint['state_dict'])  # 加载网络权重参数
    return model
