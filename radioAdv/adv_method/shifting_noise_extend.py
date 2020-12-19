'''
Author: your name
Date: 2020-12-19 13:00:55
LastEditTime: 2020-12-19 13:29:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/adv_method/shifting_noise.py
'''
import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod
import pickle
import os
import random

class Shifting_Noise_Extend(BaseMethod):
    """[BIM]
    
    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
    """
    def __init__(self, model, criterion = None, use_gpu = False, device_id = [0]):
        """[summary]

        Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
        """
        super(Shifting_Noise_Extend,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)

    def attack(self, x, y=0, x_snr=[], eps=0.03, epoch=5, is_target=False, target=0, mu = 1, shift = 20):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.
            eps (float, optional): [控制BIM精度]. Defaults to 0.03.
            epoch (int, optional): [BIM的迭代次数]. Defaults to 5.
            is_target (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.

        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        self.model.train()
        if is_target:
            x_adv,pertubation = self._attackWithTarget(x, x_snr, target, epoch, eps, mu, shift)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, x_snr, epoch, eps, mu, shift)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message
        self.model.eval()
        x_adv = torch.tensor(x_adv).to(self.device).float()
        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, x_snr, epoch, eps, mu, shift):
        
        shape = list(x.shape)
        shape[-1] = shape[-1] + shift * 2       # 在Noise两侧扩充shift
        pertubation = torch.Tensor(np.random.uniform(-eps, eps, shape)).to(self.device)      # 噪声初始化

        # dirname = './tmp_parameters'
        # parameter_name = 'Shifting_Noise_Extend_Pertubation_1.p'

        
        # if not os.path.exists(os.path.join(dirname, parameter_name)):
        #     x_list = []
        #     y_list = []
        #     x_snr_list = []
        #     pretubation_mean_list = []
        #     pretubation_list_list = []
        # else:
        #     x_list, y_list, x_snr_list, pretubation_list_list = pickle.load(open(os.path.join(dirname, parameter_name), 'rb'))
        
        # for shift in range(128):
        #     mask_shifting = np.ones(x.shape)
        #     mask_shifting[:,:,:,:shift] = 0
        #     mask_shifting = torch.tensor(mask_shifting).to(self.device)

        #     x_adv = x + torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x) * mask_shifting

        g = 0
        x_adv = x
        for i in range(epoch):
            random_shift = np.random.randint(0, 2*shift + 1)
            random_pertubation = pertubation[:, :, :, random_shift : random_shift + 128]
            x_adv = x_adv.detach() + random_pertubation
            
            x_adv.requires_grad = True
            logits = self.model(x_adv.float())
                
            loss = self.criterion(logits, y)
            self.model.zero_grad()
            loss.backward()
            
            data_grad = x_adv.grad.data
            g = mu * g + data_grad/torch.norm(data_grad, p=1)
            #得到梯度的符号
            sign_data_grad = g.sign()

            pertubation[:, :, :, random_shift : random_shift + 128] += eps * sign_data_grad
            

        x_adv = []
        for i in range(x.shape[0]):
            random_shift = np.random.randint(0, 2*shift + 1)
            random_pertubation = pertubation[i, :, :, random_shift : random_shift + 128]
            x_adv = x + random_pertubation


        x_adv = np.stack(x_adv.detach().cpu().numpy())

        pertubation = pertubation.detach().cpu().numpy()

        return x_adv, pertubation

    def _attackWithTarget(self, x, x_snr, target, epoch, eps, mu, shift):
        target = torch.tensor([target]*x.shape[0]).to(self.device)
        x_adv = x + torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x)
        g = 0
        for i in range(epoch):
            x_adv.requires_grad = True
            logits = self.model(x_adv)
                
            loss = self.criterion(logits, target)
            self.model.zero_grad()
            # if x_adv.grad is not None:
            #     x_adv.grad.data.fill_(0)
            loss.backward()

            data_grad = x_adv.grad.data
            g = mu * g + data_grad/torch.norm(data_grad, p=1)
            #得到梯度的符号
            sign_data_grad = g.sign()

            x_adv = x_adv.detach() - eps * sign_data_grad
            # x_adv = torch.clamp(x_adv, 0, 1)
            pertubation = x_adv - x

        return x_adv, pertubation