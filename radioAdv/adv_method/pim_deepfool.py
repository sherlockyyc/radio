'''
Author: your name
Date: 2020-12-21 16:56:20
LastEditTime: 2021-02-09 20:32:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/adv_method/shifting_sample.py
'''
import numpy as np
import torch
import torch.nn as nn
from adv_method.base_method import BaseMethod

class PIM_DeepFool(BaseMethod):
    """[BIM]
    
    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
    """
    def __init__(self, model, criterion = None, use_gpu = False, device_id = [0], max_iter=10, shift = 20, sample_num = 10, eps = 0.2, is_target=False, target=0):
        """[summary]

        Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
            eps (float, optional): [控制BIM精度]. Defaults to 0.03.
            epoch (int, optional): [BIM的迭代次数]. Defaults to 5.
            is_target (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.
        """
        super(PIM_DeepFool,self).__init__(model = model, criterion= criterion, use_gpu= use_gpu, device_id= device_id)
        self.max_iter = max_iter
        self.eps = eps
        self.is_target = is_target
        self.target = target
        self.shift = shift
        self.sample_num = sample_num

    def attack(self, x, y=0, x_snr=[]):
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
            x_adv,pertubation = self._attackWithTarget(x, self.target, self.max_iter, self.shift, self.sample_num )
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, self.max_iter, self.shift, self.sample_num, self.eps )
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message
        self.model.eval()
        logits = self.model(x_adv).cpu().detach().numpy()
        pred = logits.argmax(1)

        x_adv = x_adv.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        return x_adv, pertubation, logits, pred


    def _attackWithNoTarget(self, x, y, max_iter, shift, sample_num, eps ):
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
            
            sample_pertubation = torch.Tensor(np.zeros(x_adv.shape)).type_as(x_adv)
            for j in range(sample_num):
                shift_list = np.arange(-shift, shift+1)
                shift_num = np.random.choice(shift_list)
                if shift_num == 0:
                    x_new = x_adv.clone()
                else:
                    x_tail = x_adv[:,:,:, :shift_num].cpu()
                    x_head = x_adv[:,:,:, shift_num:].cpu()
                    x_new = torch.cat((x_head, x_tail), axis = -1).to(self.device).detach()
                # print(j)
                adv = x_new.clone()
                # pertubation = torch.Tensor(np.zeros(x_adv.shape)).type_as(x_adv)
                for i in range(max_iter):
                    adv.requires_grad = True
                    # 得到需要进行计算的信息
                    pred = self.model(adv)[0]
                    _,classes_now = torch.max(pred, 0)
                    pred_origin = pred[classes_origin]
                    grad_origin = torch.autograd.grad(pred_origin, adv, retain_graph= True, create_graph= True)[0]
                    # 一旦标签发生变化，退出迭代过程
                    # if classes_now != classes_origin:
                    #     break
                    # 
                    l = classes_origin
                    l_value = np.inf
                    l_w = None
                    # 遍历每一个标签，找到最近的标签，朝该方向移动
                    for k in range(classes_num):
                        if k == classes_origin:
                            continue
                        pred_k = pred[k]
                        grad_k = torch.autograd.grad(pred_k, adv, retain_graph= True, create_graph= True)[0]
                        w_k = grad_k - grad_origin
                        f_k = pred_k - pred_origin
                        value = torch.abs(f_k)/(torch.norm(w_k)**2)
                        if value < l_value :
                            l_value = value
                            l = k
                            l_w = w_k
                    # 计算扰动值
                    r = (1+0.01)*l_value * l_w
                    adv = adv + r
                    adv = adv.detach()
                    torch.cuda.empty_cache()

                
                pertubation = adv - x_new
                sample_pertubation += pertubation
                torch.cuda.empty_cache()

            sample_pertubation = sample_pertubation/sample_num
            x_adv = image + sample_pertubation

            x_advs.append(x_adv)
            pertubations.append(pertubation)
        
        # 把一个batch的图片整合起来
        x_advs = torch.cat(x_advs, dim = 0)
        pertubations = torch.cat(pertubations, dim = 0)
        # pertubations = self.norm_l1(pertubations.detach().cpu().numpy(), eps)
        # pertubations = torch.tensor(pertubations).type_as(x)
        x_advs = x + pertubations

        return x_advs, pertubations

    def _attackWithTarget(self, x, target, max_iter, shift, sample_num ):

        raise NotImplementedError
        

    # def _attackWithNoTarget(self, x, y, max_iter, shift, sample_num, eps ):
    #     x_advs = []
    #     pertubations = []
    #     for b in range(x.shape[0]):
    #         # 选择一张图片来进行查找（DeepFool目前只能一张一张的送入）
    #         img = x[b:b+1, :, :, :]
    #         x_adv = img.clone()
    #         x_adv.requires_grad = True

    #         output = self.model(x_adv)[0]

    #         _, first_predict = torch.max(output, 0)
    #         first_max = output[first_predict]
    #         grad_first = torch.autograd.grad(first_max, x_adv)[0]

    #         num_classes = len(output)

    #         sample_pertubation = torch.Tensor(np.zeros(x_adv.shape)).type_as(x_adv)
    #         for j in range(sample_num):
    #             shift_list = np.arange(-shift, shift+1)
    #             shift_num = np.random.choice(shift_list)
    #             if shift_num == 0:
    #                 x_new = x_adv.clone()
    #             else:
    #                 x_tail = x_adv[:,:,:, :shift_num].cpu()
    #                 x_head = x_adv[:,:,:, shift_num:].cpu()
    #                 x_new = torch.cat((x_head, x_tail), axis = -1).to(self.device).detach()

    #             adv = x_new.clone()
    #             for _ in range(max_iter):
    #                 adv.requires_grad = True
    #                 output = self.model(adv)[0]
    #                 _, predict = torch.max(output, 0)

    #                 # if predict != first_predict:
    #                 #     img = torch.clamp(img, min=0, max=1).detach()
    #                 #     break

    #                 r = None
    #                 min_value = None

    #                 for k in range(num_classes):
    #                     if k == first_predict:
    #                         continue

    #                     k_max = output[k]
    #                     grad_k = torch.autograd.grad(
    #                         k_max, adv, retain_graph=True, create_graph=True)[0]

    #                     prime_max = k_max - first_max
    #                     grad_prime = grad_k - grad_first
    #                     value = torch.abs(prime_max)/torch.norm(grad_prime)

    #                     if r is None:
    #                         r = (torch.abs(prime_max)/(torch.norm(grad_prime)**2))*grad_prime
    #                         min_value = value
    #                     else:
    #                         if min_value > value:
    #                             r = (torch.abs(prime_max)/(torch.norm(grad_prime)**2))*grad_prime
    #                             min_value = value

    #                 adv = adv + r
    #                 adv = adv.detach()

    #             pertubation = adv - x_new
    #             sample_pertubation += pertubation

    #         sample_pertubation = sample_pertubation/sample_num
    #         x_adv = x_adv + sample_pertubation

    #         x_advs.append(x_adv)
    #         pertubations.append(pertubation)

    #     x_advs = torch.cat(x_advs, dim = 0)
    #     pertubations = torch.cat(pertubations, dim = 0)

    #     # pertubations = self.norm_l1(pertubations.detach().cpu().numpy(), eps)
    #     # pertubations = torch.tensor(pertubations).type_as(x)
    #     x_advs = x + pertubations

    #     return x_advs, pertubations