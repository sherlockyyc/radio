import numpy as np
import torch
import torch.nn as nn


class BaseMethod(object):
    """[BaseMethod,攻击方法的基础类]

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
        super(BaseMethod,self).__init__()
        self.model = model
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.device_ids = device_id

        #------------------------------------GPU配置
        self.device = torch.device('cpu')
        if self.use_gpu:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.model = self.model.to(self.device)
                if len(self.device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=self.device_ids)

    def attack(self, x):
        """[对抗攻击，包括目标攻击和无目标攻击，可针对对应方法进行拓展]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            
        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        raise NotImplementedError
        
        # return x_adv.cpu().detach().numpy(), pertubation.cpu().detach().numpy(), pred.cpu().detach().numpy()


    def _attackWithNoTarget(self):
        """[无目标攻击方法]

        Raises:
            NotImplementedError: [description]
        """
        
        raise NotImplementedError


    def _attackWithTarget(self):
        """[目标攻击方法]

        Raises:
            NotImplementedError: [description]
        """

        raise NotImplementedError

    def into_cuda(self, data):
        """[将数据送入GPU中]

        Args:
            data ([type]): [任意可送入GPU的类型数据]

        Returns:
            data [type]: [将GPU格式的数据返回]
        """
        if self.use_gpu:
            data = data.cuda(self.device_id[0])
        return data



    def compute_jacobian(self, x, logits):
        """[根据模型的输入输出来计算jacobian矩阵]

        Args:
            x ([四维矩阵(batch, Depth, Width, Height)]): [模型的输入]
            logits ([二维矩阵(batch, class_num)]): [模型的输出]
        """
        num_classes = logits.shape[0]
        # 为每一个类别构建jacobian矩阵，初始化
        jacobian = torch.zeros(num_classes, *x.shape)
        # 构建梯度输出的mask
        grad_output = torch.zeros(*logits.shape)
