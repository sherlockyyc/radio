import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel

class Based_Transformer_Baseline(BaseModel):
    def __init__(self, output_dim):
        super(Based_Transformer_Baseline, self).__init__()
            
        # input(batch, 1, 2, 128)
        # after shape(batch, 2, 128)
        self.input_feature1 = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels= 2, out_channels= 128, kernel_size= 1, stride= 1, padding= 0),
        )
        # # after input_feature1(batch, 32, 128)
        self.input_feature2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size= 1, stride= 1, padding= 0),
        )
        # after input_feature2(batch, 64, 128)
        # after shape(batch, 128, 64)

        # input(batch, 1, 2, 128)
        # self.convtranspose1 = nn.ConvTranspose2d(in_channels= 1, out_channels= 3, kernel_size=(2,1), stride=(8,1))
        # after convtranspose1 (batch, 3, 10, 128)
        # after shape (batch, 128, 10*3)

        self.transformer_layer1 = nn.TransformerEncoderLayer(d_model= 256, nhead = 4)
        self.transformer_encoder1 = nn.TransformerEncoder(self.transformer_layer1, num_layers= 2)
        # after transformer1(batch, 128, 48)
        self.transformer_layer2 = nn.TransformerEncoderLayer(d_model= 256, nhead = 4)
        self.transformer_encoder2 = nn.TransformerEncoder(self.transformer_layer2, num_layers= 2)
        # after transformer2(batch, 128, 48)
        # 取最后一个单元的输出(batch, 1, 48), 取所有的max pool(batch, 1, 128), 取所有的Avg pool(batch, 1, 128) 
        self.max_pool = nn.MaxPool1d(kernel_size= 256)
        self.avg_pool = nn.AvgPool1d(kernel_size= 256)
        # after concat(batch, 64 + 128 + 128)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 256*128 + 128 + 128, out_features= 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= output_dim)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 2, 128)
        # x = self.convtranspose1(x)
        # x = x.transpose(1,3)
        # x = x.reshape(x.shape[0], 128, 30)
        # print(x.shape)
        x = self.input_feature1(x)
        x = self.input_feature2(x)
        x = x.transpose(1,2)
        # x = torch.cat([x] * 32, axis = -1)
        
        x = self.transformer_encoder1(x)
        x = self.transformer_encoder2(x)

        max_pool_output = self.max_pool(x).view(x.shape[0], -1)
        avg_pool_output = self.avg_pool(x).view(x.shape[0], -1)
        # last_layer_output = x[:,-1, :]
        last_layer_output = x.view(x.shape[0], -1)
        # print(max_pool_output.shape)
        # print(avg_pool_output.shape)
        # print(last_layer_output.shape)
        x = torch.cat([last_layer_output, max_pool_output, avg_pool_output], axis = -1)
        
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x