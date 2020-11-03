import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class DeepFool(BaseMethod):
    """[DeepFool]

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
        super(DeepFool,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)

    def attack(self, x, y=0, max_iter=10, is_target=False, target=0):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.
            max_iter (int, optional): [DeepFool的最大迭代次数]. Defaults to 10.
            is_target (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.
            
        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        if is_target:
            x_adv,pertubation = self._attackWithTarget(x, target)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, max_iter)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message

        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, max_iter):
        
        x_advs = []
        pertubations = []
        for b in range(x.shape[0]):
            # 选择一张图片来进行查找（DeepFool目前只能一张一张的送入）
            image = x[b:b+1, :, :, :]
            x_adv = image

            # 得到原始的预测结果
            predict_origin = self.model(x_adv).cpu().detach().numpy()
            # 所有的类别数目
            classes_num = predict_origin.shape[1]
            # 得到原始的分类结果
            classes_origin = np.argmax(predict_origin, axis= 1)[0]
            
            for i in range(max_iter):
                x_adv.requires_grad = True
                # 得到需要进行计算的信息
                pred = self.model(x_adv)[0]
                _,classes_now = torch.max(pred, 0)
                pred_origin = pred[classes_origin]
                grad_origin = torch.autograd.grad(pred_origin, x_adv, retain_graph= True, create_graph= True)[0]
                # 一旦标签发生变化，退出迭代过程
                if classes_now != classes_origin:
                    break
                # 
                l = classes_origin
                l_value = np.inf
                l_w = None
                # 遍历每一个标签，找到最近的标签，朝该方向移动
                for k in range(classes_num):
                    if k == classes_origin:
                        continue
                    pred_k = pred[k]
                    grad_k = torch.autograd.grad(pred_k, x_adv, retain_graph= True, create_graph= True)[0]
                    w_k = grad_k - grad_origin
                    f_k = pred_k - pred_origin
                    value = torch.abs(f_k)/(torch.norm(w_k)**2)
                    if value < l_value :
                        l_value = value
                        l = k
                        l_w = w_k
                # 计算扰动值
                r = (1+0.02)*l_value * l_w
                x_adv = x_adv + r
                x_adv = x_adv.detach()
                x_adv = torch.clamp(x_adv, 0, 1)
                

            pertubation = x_adv - image


            x_advs.append(x_adv)
            pertubations.append(pertubation)
        
        # 把一个batch的图片整合起来
        x_advs = torch.cat(x_advs, dim = 0)
        pertubations = torch.cat(pertubations, dim = 0)

        return x_advs, pertubations

    def _attackWithTarget(self, x, target):
        
        raise NotImplementedError