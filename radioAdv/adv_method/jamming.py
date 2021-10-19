import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class Jamming(BaseMethod):
    """[FGSM]

    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
        self.use_gpu ([bool]): 是否使用GPU
        self.device_id ([list]): 使用的GPU的id号
    """
    def __init__(self, model, criterion = None, use_gpu = False, device_id = [0], mean = 0.2, std = 0.1):
        """[summary]

       Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
            use_gpu ([bool]): 是否使用GPU
            device_id ([list]): 使用的GPU的id号
        """
        super(Jamming,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)
        self.mean = mean
        self.std = std



    def attack(self, x, y=0):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.
            mean (float): jamming 生成的噪声信号的均值
            std (float):        jamming 生成的噪声信号的标准差
            
        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        
        
        self.model.eval()

        x_adv,pertubation = self._attackWithNoTarget(x, y, self.mean, self.std)

        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, mean, std):
        
        # pertubation = np.random.normal(mean, std, x.shape)
        # print(np.max(pertubation))
        pertubation = np.random.randn(*x.shape) * mean
        pertubation = torch.tensor(pertubation).to(self.device)
        
        x_adv = x + pertubation
        x_adv = x_adv.float()
        return x_adv, pertubation



    
    
