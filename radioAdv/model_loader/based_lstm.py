import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_loader.base_model import BaseModel

class Based_LSTM(BaseModel):
    def __init__(self, output_dim):
        super(Based_LSTM, self).__init__()
        # input(batch, 1, 2, 128)
        # after shape(batch, 128, 2)
        self.lstm1 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.LSTM(input_size= 2, hidden_size= 100, num_layers=3, batch_first= True)
        )
        self.relu1 = nn.ReLU()
        # after lstm1(batch, 128, 100)
        self.lstm2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.LSTM(input_size= 100, hidden_size=50, num_layers=3, batch_first= True)
        )
        self.relu2 = nn.ReLU()
        # afer lstm2(batch, 128, 50)
        # after shape(batch, 128 * 50)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 128 * 50, out_features= 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 64, out_features= output_dim)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 2, 128)
        x = x.transpose(1,2)
        x,_ = self.lstm1(x)
        x = self.relu1(x)
        # print('conv1', x.shape)
        x,_ = self.lstm2(x)
        x = self.relu2(x)
        x = x.reshape(x.shape[0], -1)
        # print('gru2', x.shape)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def loadBased_LSTM(filepath):
    """[加载LeNet网络模型]

    Args:
        filepath ([str]): [LeNet的预训练模型所在的位置]

    Returns:
        [type]: [返回一个预训练的LeNet]
    """
    checkpoint = torch.load(filepath)
    model = Based_LSTM(output_dim = 11)
    model.load_state_dict(checkpoint['state_dict'])  # 加载网络权重参数
    return model