import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class CW(BaseMethod):
    """[FGSM]

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
            device_id ([list]): 使用的GPU的id号
        """
        super(CW,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)

    def attack(self, x, y=0, x_snr=[], binary_search_steps=9, n_iters=10000, c=1e-4, kappa=0, lr=0.01, is_target=False, target=0):
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
        
        
        self.model.eval()
        if is_target:
            x_adv,pertubation = self._attackWithTarget(x, target, binary_search_steps, n_iters, c, kappa, lr)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, binary_search_steps, n_iters, c, kappa, lr)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message
        self.model.eval()

        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)


        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, binary_search_steps, n_iters, c, kappa, lr):
        x_arctanh = self.arctanh(x)

        for _ in range(binary_search_steps):
            delta = torch.zeros_like(x).type_as(x)
            delta.detach_()
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=lr)
            prev_loss = 1e6

            for step in range(n_iters):
                optimizer.zero_grad()
                adv_examples = self.scaler(x_arctanh + delta)
                loss1 = torch.sum(c * self._f(adv_examples, y, kappa))
                loss2 = torch.functional.F.mse_loss(adv_examples, x, reduction='sum')

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                if step % (n_iters // 10) == 0:
                    if loss > prev_loss:
                        break

                    prev_loss = loss

            x_adv = self.scaler(x_arctanh + delta).detach()
            pertubation = x_adv - x
            
            pertubation = self.norm_l1(pertubation.detach().cpu().numpy(), eps)
            pertubation = torch.tensor(pertubation).type_as(x)
            x_adv = x + pertubation
            return x_adv, pertubation

    def _attackWithTarget(self, x, target, binary_search_steps, n_iters, c, kappa, lr):
        
        return None, None


    def _f(self, adv_imgs, labels, kappa):
        outputs = self.model(adv_imgs)
        y_onehot = torch.nn.functional.one_hot(labels, num_classes=11)

        real = (y_onehot * outputs).sum(dim=1)
        other, _ = torch.max((1-y_onehot)*outputs, dim=1)

        loss = torch.clamp(real-other, min=-kappa)

        return loss

    def arctanh(self, imgs):
        scaling = torch.clamp(imgs, max=1, min=-1)
        x = 0.999999 * scaling

        return 0.5*torch.log((1+x)/(1-x))

    def scaler(self, x_atanh):
        return ((torch.tanh(x_atanh))+1) * 0.5

    
    
