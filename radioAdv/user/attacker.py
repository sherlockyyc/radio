import os
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

import utils as util
from config import *

class Attacker(object):
    """[攻击者]

    Args:
        self.model ([]): [要攻击的模型]
        self.criterion ([]): 损失函数
        self.config ([]): 配置类
        self.attack_method ([]): 攻击方法
        self.use_gpu ([bool]): 是否使用GPU
        self.device_ids ([list]): 使用的GPU的id号
        self.attack_name ([str]): 攻击方法名称
        self.is_target ([bool]): 是否进行目标攻击
        self.target ([int]): 目标攻击的目标 
    """
    def __init__(self,model,criterion,config,attack_method, snrs = [], mods = []):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.attack_method = attack_method
        self.snrs = snrs
        self.mods = mods
        #########################GPU配置
        self.use_gpu = False
        self.device_ids = self.config.GPU['device_id']
        if self.device_ids:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                assert len(self.device_ids) > 0,message
                self.model = self.model.cuda(self.device_ids[0])
                if len(self.device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=self.device_ids)
                self.use_gpu = True
        #########################攻击信息
        self.attack_name = self.config.CONFIG['attack_name']
        #########################攻击方式---目标攻击设置
        self.is_target = False
        self.target = 0
        if 'is_target' in getattr(self.config,self.attack_name):
            self.is_target = getattr(self.config,self.attack_name)['is_target']
            self.target =  getattr(self.config,self.attack_name)['target']        

        # 各种路径配置
        #------------------------------------figure配置
        self.figure_dir = os.path.join('figure', self.config.CONFIG['model_name'] )


    def attack_batch(self, x, y):
        """[对一组数据进行攻击]]

        Args:
            x ([array]): [一组输入数据]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_advs ([array]): [得到的对抗样本，四维]
            pertubations ([array]): [得到的对抗扰动，四维]
            nowLabels ([array]): [攻击后的样本标签，一维]
        """ 
        
        #放到GPU设备中
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        if type(y) is not torch.tensor:
            y = torch.Tensor(y.float())
        if self.use_gpu:
            x = x.cuda(self.device_ids[0]).float()
            y = y.cuda(self.device_ids[0]).long()

        x_advs, pertubations, logits, nowLabels = self.attack_method.attack(
            x, y, **getattr(self.config, self.config.CONFIG['attack_name']))

        return x_advs, pertubations, logits, nowLabels 


    def attack_one_img(self, x, y=[0]):   
        """[攻击一个图片]

        Args:
            x ([array]): [一个输入数据]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_adv ([array]): [得到的对抗样本，三维]
            pertubation ([array]): [得到的对抗扰动，三维]
            nowLabel ([int]): [攻击后的样本标签]
        """
        x = np.expand_dims(x, axis=0)                    #拓展成四维
        y = np.array(list([y]))                         #转成矩阵
        x_adv, pertubation, logits, nowLabel = self.attack_batch(x, y)
        return x_adv[0], pertubation[0], logits[0], nowLabel[0]


    def attack_set(self, data_loader):
        """[对一个数据集进行攻击]

        Args:
            data_loader ([DataLoader]): [数据加载器]

        Returns:
            acc [float]: [攻击后的准确率]
            mean [float]: 平均扰动大小
        """
        log = {}

        success_num = 0
        data_num = 0
        pertubmean = []

        pertub_num = 0
        pertub_sum = 0

        pertub_max = 0

        predict = []
        targets = []
        snr_acc = np.zeros(len(self.snrs))
        snr_num = np.zeros(len(self.snrs))

        for idx,(x,y, x_snr) in enumerate(tqdm(data_loader)):
            x_advs ,pertubations, logits, nowLabels = self.attack_batch(x, y)
            # print(x[0])
            # break
            y = y.detach().cpu().numpy()
            # 计算平均攻击成功率, 
            if self.is_target:
                success_num += ((self.target == nowLabels) == (self.target != y)).sum()
                data_num += (self.target != y).sum()
            else:
                data_num +=  x.shape[0]
                success_num += (y != nowLabels).sum()

            # 统计平均扰动值
            pertubmean.append(pertubations.mean())

            # 计算平均扰动比例            
            pertub_sum += np.sum(pertubations != None)
            pertub_num += np.sum(pertubations != None) - np.sum(np.abs(pertubations) < 1e-3)
            # pertub_num += np.sum(np.abs(x.numpy()) < 1e-5)

            # 计算最大扰动值
            if np.max(np.abs(pertubations)) > pertub_max:
                pertub_max = np.max(np.abs(pertubations))

            # 保存预测结果
            predict.append(logits)
            targets.append(y)
            # 计算各snr下的准确率
            x_snr = x_snr.numpy()
            for i, snr in enumerate(self.snrs):
                if np.sum(x_snr== snr) != 0:
                    snr_acc[i] += np.sum(nowLabels[x_snr == snr] == y[x_snr == snr])
                    snr_num[i] += np.sum(x_snr == snr)

        # 记录不同snr下的准确率
        snr_acc = snr_acc / snr_num
        self.plot_snr_figure(self.figure_dir, snr_acc, self.snrs)
        # 绘制混淆矩阵
        predict = np.vstack(predict)
        targets = np.hstack(targets)
        self.plot_confusion_matrix_figure(self.figure_dir, predict, targets, self.mods)
        # 记录攻击结果
        mean = np.mean(pertubmean)
        acc = 1 - success_num / data_num
        pertub_prop = pertub_num / pertub_sum
        log['acc'] = acc
        log['pertubmean'] = mean
        log['pertub_prop'] = pertub_prop
        log['pertub_max'] = pertub_max
        log['snr_acc'] = snr_acc
        return log
  

    def plot_snr_figure(self, dirname, snr_acc, snrs):
        """[基于不同SNR下的分类准确率绘制图像]

        Args:
            dirname ([str]): [存储图像的文件夹]
            snr_acc ([一维array]]): [不同SNR下的分类准确率]
            snrs ([一维array]): [不同的SNR的名称]
        """
        # Plot accuracy curve
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%m-%d-%H:%M")

        util.ensure_dir(dirname)
        plt.plot(snrs, snr_acc)
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy On Different SNR")
        plt.savefig(os.path.join(dirname, 'Classification Accuracy On Different SNR' + now_time + '.jpg'))
        print("Figure 'Classification Accuracy On Different SNR' generated successfully")


    def plot_confusion_matrix_figure(self, dirname, predict, targets, mods):
        """[绘制预测结果的混淆矩阵]

        Args:
            dirname ([str]): [存储图像的文件夹]
            predict ([二维array,(length, probality)]]): [网络的得到预测值]]
            targets ([一维array 或 二维array（onehot）]): [对应的真实标签]
            mods ([一维array]): 真实类别，str
        """
        cm = util.generate_confusion_matrix(predict, targets, mods)
        util.ensure_dir(dirname)
        util.plot_confusion_matrix(cm, dirname, mods)
        print("Figure 'Confusion Matrix' generated successfully")
        