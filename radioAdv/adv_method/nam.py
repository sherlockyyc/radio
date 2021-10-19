'''
Author: your name
Date: 2021-02-08 18:56:51
LastEditTime: 2021-02-08 20:10:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/adv_method/nam.py
'''
import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class NAM(BaseMethod):
    """[BIM]
    
    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
    """
    def __init__(self, model, criterion = None, use_gpu = False, device_id = [0], eps=0.03, epoch=5, beta1 = 0.9, beta2 = 0.999, is_target=False, target=0):
        """[summary]

        Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
            eps (float, optional): [控制BIM精度]. Defaults to 0.03.
            epoch (int, optional): [BIM的迭代次数]. Defaults to 5.
            is_target (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.
        """
        super(NAM,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)
        self.eps = eps
        self.epoch = epoch
        self.beta1 = beta1
        self.beta2 = beta2
        self.is_target = is_target
        self.target = target

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
            x_adv,pertubation = self._attackWithTarget(x, self.target, self.epoch)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, self.epoch, self.eps, self.beta1, self.beta2)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message
        self.model.eval()
        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, epoch, eps, beta1, beta2):
        x_adv = x + torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x)
        g = 0
        m = 0
        n = 0
        for i in range(epoch):
            # x_nes = x_adv + eps * mu * g
            x_adv.requires_grad = True
            logits = self.model(x_adv)
                
            loss = self.criterion(logits, y)
            self.model.zero_grad()
            loss.backward()

            g = x_adv.grad.data
            g_s = g/(1 - beta1 ** i)
            m = beta1 * m + (1 - beta1) * g
            m_s = m/(1 - beta1 ** i)
            n = beta2 * n + (1 - beta2) * g * g
            n_s = n/(1 - beta2 ** i)
            m_b = beta1 * m_s + (1 - beta1) * g_s
            
            r = m_b / (torch.sqrt(n_s + 1e-8))
            #得到梯度的符号
            sign_data_grad = r.sign()

            x_adv = x_adv.detach() + eps * sign_data_grad
            # x_adv = torch.clamp(x_adv, 0, 1)
        pertubation = x_adv - x
        pertubation = self.norm_l1(pertubation.detach().cpu().numpy(), epoch * eps)
        pertubation = torch.tensor(pertubation).type_as(x)
        x_adv = x + pertubation
        return x_adv, pertubation

    def _attackWithTarget(self, x, target, epoch):

        raise NotImplementedError