'''
Author: your name
Date: 2020-12-21 16:56:20
LastEditTime: 2021-02-08 19:55:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/adv_method/shifting_sample.py
'''
import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class PIM_NAM(BaseMethod):
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
        super(PIM_NAM,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)

    def attack(self, x, y=0, x_snr=[], eps=0.03, epoch=5, is_target=False, target=0, beta1 = 0.9, beta2 = 0.999, shift = 20, sample_num = 10):
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
        self.model.eval()
        if is_target:
            x_adv,pertubation = self._attackWithTarget(x, target)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, epoch, eps, beta1, beta2, shift, sample_num )
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message
        self.model.eval()
        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, epoch, eps, beta1, beta2, shift, sample_num ):
        pertubation = torch.Tensor(np.random.uniform(-eps, eps, x.shape)).type_as(x)
        x_adv = x
        g = 0
        m = 0
        n = 0
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

                g = x_new.grad.data
                g_s = g/(1 - beta1 ** i)
                m = beta1 * m + (1 - beta1) * g
                m_s = m/(1 - beta1 ** i)
                n = beta2 * n + (1 - beta2) * g * g
                n_s = n/(1 - beta2 ** i)
                m_b = beta1 * m_s + (1 - beta1) * g_s
                
                r = m_b / (torch.sqrt(n_s + 1e-8))
                #得到梯度的符号
                sign_data_grad = r.sign()
                sample_pertubation += eps * sign_data_grad
                
            sample_pertubation = sample_pertubation/sample_num
            pertubation += sample_pertubation
        
        pertubation = self.norm_l1(pertubation.detach().cpu().numpy(), epoch * eps)
        pertubation = torch.tensor(pertubation).type_as(x)
        x_adv = x + pertubation

        return x_adv, pertubation

    def _attackWithTarget(self, x, target ):
        
        raise NotImplementedError