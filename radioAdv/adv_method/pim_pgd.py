'''
Author: your name
Date: 2020-12-21 16:56:20
LastEditTime: 2021-02-08 20:06:20
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/adv_method/shifting_sample.py
'''
import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class PIM_PGD(BaseMethod):
    """[BIM]
    
    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
    """
    def __init__(self, model, criterion = None, use_gpu = False, device_id = [0], eps=0.03, epoch=5, shift = 20, sample_num = 10, is_target=False, target=0):
        """[summary]

        Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
        """
        super(PIM_PGD,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)
        self.eps = eps
        self.epoch = epoch
        self.is_target = is_target
        self.target = target
        self.shift = shift
        self.sample_num = sample_num

    def attack(self, x, y=0):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.

        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        self.model.eval()
        if self.is_target:
            x_adv,pertubation = self._attackWithTarget(x, self.target, self.epoch, self.eps, self.shift, self.sample_num )
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, self.epoch, self.eps, self.shift, self.sample_num )
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message
        self.model.eval()
        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, epoch, eps, shift, sample_num ):
        pertubation = torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x)
        x_adv = x
        g = 0
        for i in range(epoch):
            x_adv = x + pertubation
            sample_pertubation = torch.Tensor(np.zeros(x.shape)).type_as(x)
            for j in range(sample_num):
                shift_list = np.arange(-shift, shift+1)
                shift_num = np.random.choice(shift_list)
                if shift_num == 0:
                    x_new = x_adv
                else:
                    x_tail = x_adv[:,:,:, :shift_num].cpu()
                    x_head = x_adv[:,:,:, shift_num:].cpu()
                    x_new = torch.cat((x_head, x_tail), axis = -1).to(self.device).detach()

                x_new.requires_grad = True

                logits = self.model(x_new)
                
                loss = self.criterion(logits, y)
                self.model.zero_grad()
                loss.backward()

                data_grad = x_new.grad.data
                #得到梯度的符号
                sign_data_grad = data_grad.sign()
                sample_pertubation += eps * sign_data_grad
                
            sample_pertubation = sample_pertubation/sample_num
            pertubation += sample_pertubation
        
        pertubation = self.norm_l1(pertubation.detach().cpu().numpy(), epoch * eps)
        pertubation = torch.tensor(pertubation).type_as(x)
        
        x_adv = x + pertubation

        return x_adv, pertubation

    def _attackWithTarget(self, x, target, epoch, eps, shift, sample_num ):

        raise NotImplementedError