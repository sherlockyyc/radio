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
    train_dataset = getattr(module_loader, config.CONFIG['dataset_name']+'TrainSet')(**getattr(config, config.CONFIG['dataset_name']))    
    train_loader = DataLoader(train_dataset, batch_size = config.ARG['batch_size'], shuffle=True, num_workers=4)
    test_dataset = getattr(module_loader, config.CONFIG['dataset_name']+'TestSet')(**getattr(config, config.CONFIG['dataset_name']))  
    snrs, mods = test_dataset.get_snr_and_mod()  
    test_loader = DataLoader(test_dataset, batch_size = config.ARG['batch_size'], shuffle=False, num_workers=1)
    
    metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

    model = getattr(module_model, config.CONFIG['model_name'])(**getattr(config,config.CONFIG['model_name']))
    criterion = None

    optimizer = None

    trainer = module_trainer.ClassficationTrainer(model, train_loader, test_loader, criterion, optimizer, metrics, config, snrs = snrs, mods = mods)
    print('start test')
    trainer.test()

