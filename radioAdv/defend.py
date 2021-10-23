import torch.nn as nn
from torch.utils.data import DataLoader

import data_loader as module_loader
import model_loader as module_model
import user as module_trainer
import metric as module_metric
from defend_config import Defend_Config
import defender  as module_defender
from utils import *
import argparse
import yaml

if __name__ == '__main__':
    setup_seed(2000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        default=None,
                        help="The config parameter yaml file to be changed",
                        type=str)

    parser.add_argument('--vGPU',
                        nargs='+',
                        type=int,
                        default=None,
                        help="Specify which GPUs to use.")

    args = parser.parse_args()


    config = Defend_Config()

    if args.config:
        assert os.path.exists(args.config), "There's no '" + args.config + "' file."
        with open(args.config, "r") as load_f:
            config_parameter = yaml.load(load_f)
            config.load_parameter(config_parameter)

    if args.vGPU:
        config.GPU['device_id'] = args.vGPU
        
    Log = config.log_output()

    train_dataset = getattr(module_loader, config.CONFIG['dataset_name']+'TrainSet')(**getattr(config, config.CONFIG['dataset_name']))    
    train_loader = DataLoader(train_dataset, batch_size = config.ARG['batch_size'], shuffle=True, num_workers=4)
    test_dataset = getattr(module_loader, config.CONFIG['dataset_name']+'TestSet')(**getattr(config, config.CONFIG['dataset_name']))  
    snrs, mods = test_dataset.get_snr_and_mod()  
    test_loader = DataLoader(test_dataset, batch_size = config.ARG['batch_size'], shuffle=False, num_workers=1)

    metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

    model = getattr(module_model, config.CONFIG['model_name'])(**getattr(config,config.CONFIG['model_name']))
    criterion = getattr(nn, config.CONFIG['criterion_name'])()

    optimizer = getattr(torch.optim, config.CONFIG['optimizer_name'])(model.parameters(),**getattr(config, config.CONFIG['optimizer_name']))

    
    trainer = getattr(module_defender, config.CONFIG['defender_name'])(model, train_loader, test_loader, criterion, optimizer, metrics, config, snrs = snrs, mods = mods)
    print('start train')
    trainer.train()

