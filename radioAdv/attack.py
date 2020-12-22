'''
Author: your name
Date: 2020-12-07 15:03:08
LastEditTime: 2020-12-21 20:31:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/attack.py
'''
import adv_method
import data_loader
import model_loader
from model_loader import *
from user import *
from config import *
from utils import *
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


######################################加载配置
setup_seed(2020)
config = Config()
Log = config.log_output()

######################################加载数据集
dataset_name = config.CONFIG['dataset_name']
attack_set = getattr(data_loader, dataset_name + 'AttackSet')(**getattr(config, dataset_name))
# test_set = getattr(data_loader, dataset_name + 'TestSet')(**getattr(config, dataset_name))
snrs, mods = attack_set.get_snr_and_mod()  
# snrs, mods = test_set.get_snr_and_mod()

######################################加载数据加载器
attack_loader = DataLoader(attack_set, batch_size = 32, shuffle = False, num_workers = 4)
# test_loader = DataLoader(test_set, batch_size = 32, shuffle = True, num_workers = 1)

######################################加载模型
model_name = config.CONFIG['model_name']
model = getattr(model_loader, 'load' + model_name)(**getattr(config, model_name))
model.eval()
######################################加载损失函数
criterion_name = config.CONFIG['criterion_name']
criterion = getattr(nn, criterion_name)()

## 加载metric
metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

######################################加载攻击方式
attack_name = config.CONFIG['attack_name']
attack_method = getattr(adv_method, attack_name)(model, criterion, **config.GPU)


######################################加载攻击训练器
attacker = Attacker(model, metrics, criterion, config, attack_method, snrs, mods)


#####################################开始攻击

###############################攻击一张图片
# x,y = TrainSet.__getitem__(7000)
# x_adv,pertubation,nowLabel = attacker.attack_one_img(x,y)
# print(y,nowLabel)

###############################攻击整个数据集
# attack_log = attacker.attack_set(test_loader)
# Log['attack_log'] = attack_log
log = attacker.start_attack(attack_loader)
Log.update(log)

####################################log保存
filename = os.path.join(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])
f = open(filename,'w')

for key, value in Log.items():
    print('    {:15s}: {}'.format(str(key), value))
    log = {}
    log[key] = value
    log_write(f, log)


# # plt.switch_backend('agg')
# plt.imshow(x[0])
# plt.imshow(x_adv[0])
# plt.show()
# plt.imshow(x_adv[0])

# attacker.generate_data(test_loader)
# dirname = '/home/yuzhen/wireless/RML2016.10a'
# attack_x, attack_y, attack_snr = pickle.load(open(os.path.join(dirname, 'attack_data.p'), 'rb'))
# print(len(attack_x))

# attacker.pick_data(test_loader)