'''
Author: your name
Date: 2021-02-08 18:56:51
LastEditTime: 2021-02-09 10:16:12
LastEditors: your name
Description: In User Settings Edit
FilePath: /radioClassifyFrame/adv_train.py
'''
import torch.nn as nn
from torch.utils.data import DataLoader

import data_loader as module_loader
import model as module_model
import trainer as module_trainer
import metric as module_metric
from config import Config
from utils import *


if __name__ == '__main__':
    setup_seed(2000)

    config = Config()
    dataset = getattr(module_loader, config.CONFIG['dataset_name']+'AdvTrainSet')(**getattr(config, config.CONFIG['dataset_name']))    
    data_loader = DataLoader(dataset, batch_size = config.ARG['batch_size'], shuffle=True, num_workers=4)
    metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

    model = getattr(module_model, config.CONFIG['model_name'])(**getattr(config,config.CONFIG['model_name']))
    criterion = getattr(nn, config.CONFIG['criterion_name'])()

    optimizer = getattr(torch.optim, config.CONFIG['optimizer_name'])(model.parameters(),**getattr(config, config.CONFIG['optimizer_name']))

    trainer = module_trainer.ClassficationTrainer(model, data_loader, criterion, optimizer, metrics, config)
    print('start train')
    trainer.train()

