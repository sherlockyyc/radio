import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class JSMA(BaseMethod):
    """[JSMA]

    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
        self.use_gpu ([bool]): 是否使用GPU
        self.device_id ([list]): 使用的GPU的id号
    """
    def __init__(self, model, criterion = None, use_gpu = False, device_id = [0]):
        """[summary]

       Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
            use_gpu ([bool]): 是否使用GPU
            device_ids ([list]): 使用的GPU的id号
        """
        super(JSMA,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)

    def attack(self, x, y=0, eps=0.03, is_target=False, target=0):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.
            eps (float, optional): [控制FGSM精度]. Defaults to 0.03.
            is_target (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.
            
        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        if is_target:
            x_adv,pertubation = self._attackWithTarget(x, target, eps)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, eps)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message

        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred

    def _attackWithNoTarget(self, x, y, eps):
        
        x_adv = x
        x_adv.requires_grad = True
        logits = self.model(x_adv)
            
        loss = self.criterion(logits, y)
        self.model.zero_grad()
        # if x_adv.grad is not None:
        #     x_adv.grad.data.fill_(0)
        loss.backward()
        # logits.backward(torch.ones_like(logits))

        data_grad = x_adv.grad.data

        #得到梯度的符号
        sign_data_grad = data_grad.sign()

        x_adv = x_adv + eps * sign_data_grad
        x_adv = torch.clamp(x_adv, 0, 1)
        pertubation = x_adv - x

        return x_adv, pertubation

    def _attackWithTarget(self, x, target, eps):


        target = torch.tensor([target] * x.shape[0]).cuda()
        x_adv = x
        x_adv.requires_grad = True
        logits = self.model(x_adv)
        # 计算Jacobian矩阵
        jacobian = self.compute_jacobian(x_adv, logits)


        loss = self.criterion(logits, target)
        self.model.zero_grad()
        # if x_adv.grad is not None:
        #     x_adv.grad.data.fill_(0)
        loss.backward()

        data_grad = x_adv.grad.data
        #得到梯度的符号
        sign_data_grad = data_grad.sign()

        x_adv = x_adv - eps * sign_data_grad
        x_adv = torch.clamp(x_adv, 0, 1)
        pertubation = x_adv - x

        return x_adv, pertubation

    def compute_jacobian(self, x, logits):
        """[根据模型的输入输出来计算jacobian矩阵]

        Args:
            x ([四维矩阵(batch, Depth, Width, Height)]): [模型的输入]
            logits ([二维矩阵(batch, class_num)]): [模型的输出] x.requires_grad = True
        """
        num_classes = logits.shape[0]
        # 为每一个类别构建jacobian矩阵，初始化
        jacobian = torch.zeros(num_classes, *x.shape)
        # 构建梯度输出的mask
        grad_mask = torch.zeros(*logits.shape)
        # 将数据送入GPU中
        jacobian = self.into_cuda(jacobian)
        grad_mask = self.into_cuda(grad_mask)
        # Jacobian矩阵的计算
        for i in range(num_classes):
            troch.autograd.gradcheck.zero_gradients(x)
            grad_mask.zero_()
            grad_mask[:, i] = 1
            logits.backward(grad_mask, retain_variables = True)
            jacobian[i] = x.grad.data
        
        # 将前两维换过来
        jacobian = torch.transpose(jacobian, dim0 = 0, dim1 = 1)
        return jacobian

    def compute_saliency_map(self, jacobian):
        """[根据Jacobian矩阵计算相应的saliency_map]

        Args:
            jacobian ([五维矩阵]]): [(batch, num_classes, Depth, Width, Height)]
        """
        