import torch.nn as nn
from torch.utils.data import DataLoader

import data_loader as module_loader
import model as module_model
import trainer as module_trainer
import metric as module_metric
from config import Config
from utils import *
import pickle

if __name__ == '__main__':
    setup_seed(2000)

    config = Config()
    # dataset = getattr(module_loader, config.CONFIG['dataset_name']+'TestSet')(**getattr(config, config.CONFIG['dataset_name']))  
    # snrs, mods = dataset.get_snr_and_mod()  
    # data_loader = DataLoader(dataset, batch_size = config.ARG['batch_size'], shuffle=False, num_workers=1)
    # metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]
    
    model = getattr(module_model, config.CONFIG['model_name'])(**getattr(config,config.CONFIG['model_name']))
    model_filename = '/home/yuzhen/wireless/model/VTCNN2/VTCNN2_Epoch85.pkl'
    checkpoint = torch.load(model_filename, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    pim_white_log, pim_real_sample, pim_adv_sample, pim_adv_pertub, pim_targets, pim_x_snrs = pickle.load(open('./vtcnn2_pim_mim_0030.p', 'rb'))
    index_10 = np.where(pim_x_snrs == 10)
    index = index_10[0][93]
    label = pim_targets[index]

    print('-------------original signal--------------------')
    signal = torch.tensor(pim_real_sample[index])
    signal = signal.expand(1,1,2,128)
    logits = model(signal)[0]
    prob = torch.nn.Softmax()(logits)
    print(logits)
    print(prob)
    print('label:{}, predict:{}'.format(label, np.argmax(logits.detach().numpy())))

    # adv
    print('-------------adversarial signal--------------------')
    adv_singal = torch.tensor(pim_adv_sample[index])
    signal = signal.expand(1,1,2,128)
    logits = model(signal)[0]
    prob = torch.nn.Softmax()(logits)
    print(logits)
    print(prob)
    print('label:{}, predict:{}'.format(label, np.argmax(logits.detach().numpy())))

    # criterion = None

    # optimizer = None

    # trainer = module_trainer.ClassficationTrainer(model, data_loader, criterion, optimizer, metrics, config, snrs = snrs, mods = mods)
    # print('start test')
    # trainer.test()

