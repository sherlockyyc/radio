import os
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader

import model_loader
from config import *
import utils as util
import metric as module_metric
import data_loader as module_data_loader
import pickle



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
    def __init__(self,model, metrics, criterion,config,attack_method, snrs = [], mods = []):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.attack_method = attack_method
        self.snrs = snrs
        self.mods = mods
        self.metrics = metrics


        #########################GPU配置
        self.use_gpu = False
        self.device_ids = [0]
        self.device = torch.device('cpu')
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.model = self.model.to(self.device)
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



    def attack_batch(self, model, x, y):
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
        # if type(x) is np.ndarray:
        #     x = torch.from_numpy(x)
        # if type(y) is not torch.tensor:
        #     y = torch.Tensor(y.float())
        x = torch.Tensor(x.float()).to(self.device)
        y = torch.Tensor(y.float()).to(self.device).long()


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
        x_adv,pertubation,nowLabel = self.attack_batch(self.model, x, y)
        return x[0], x_adv[0], pertubation[0] ,nowLabel[0]


    def attack_set(self, model, data_loader):
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

        real_sample = []
        adv_sample = []
        adv_pertub = []
        x_snrs = []

        snr_acc = np.zeros(len(self.snrs))
        snr_num = np.zeros(len(self.snrs))

        for idx,(x,y, x_snr) in enumerate(tqdm(data_loader)):
            x_advs ,pertubations, logits, nowLabels = self.attack_batch(model, x, y)
            y = y.detach().cpu().numpy()
            # 计算平均攻击成功率, 
            if self.is_target:
                success_num += ((self.target == nowLabels) == (self.target != y)).sum()
                data_num += (self.target != y).sum()
            else:
                data_num +=  x.shape[0]
                success_num += (y != nowLabels).sum()

            # 统计平均扰动值
            pertubmean.append(np.abs(pertubations).mean())

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

            # 用来保存信息
            real_sample.append(x)
            adv_sample.append(x_advs)
            adv_pertub.append(pertubations)
            x_snrs.append(x_snr)

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

        real_sample = np.vstack(real_sample)
        adv_sample = np.vstack(adv_sample)
        adv_pertub = np.vstack(adv_pertub)
        x_snrs = np.hstack(x_snrs)

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
        return log, real_sample, adv_sample, adv_pertub, targets, x_snrs

    def start_attack(self, data_loader):
        attack_method = self.config.Switch_Method['method']
        log = {}
        if attack_method == 'White_Attack':
            log = self.white_attack(data_loader)
        elif attack_method == 'Black_Attack':
            threat_model = self.config.Black_Attack['threat_model']
            black_model = self.config.Black_Attack['black_model']
            log = self.black_attack(data_loader, threat_model, black_model)
        elif attack_method == 'Shifting_Attack':
            threat_model = self.config.Black_Attack['threat_model']
            black_model = self.config.Black_Attack['black_model']
            log = self.shifting_attack(data_loader, threat_model, black_model)


        return log

    def white_attack(self, data_loader):
        log, _, _, _, _, _ = self.attack_set(self.model, data_loader)
        return log


    def black_attack(self, data_loader, threat_model, black_model):
        """[对一个模型进行黑盒攻击]

        Args:
            data_loader ([DataLoader]): [数据加载器]
            threat_model ([Model]): [用于生成对抗样本的白盒模型]
            black_model ([Model]): [用于攻击的黑盒模型]

        Returns:
            acc [float]: [攻击后的准确率]
            mean [float]: 平均扰动大小
        """
        log = {}

        model_name = self.config.Black_Attack['threat_model']
        white_model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)
        white_log, real_sample, adv_sample, adv_pertub, targets, x_snrs = self.attack_set(white_model, data_loader)
        
        for key, value in white_log.items():
            log['White_model   '+key] = value


        torch.cuda.empty_cache()

        model_name = self.config.Black_Attack['black_model']
        black_model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)

        black_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample, targets, x_snrs)
        adv_loader = DataLoader(black_dataset, batch_size = 32, shuffle = False, num_workers = 4)

        black_log = self._model_test(black_model, adv_loader)


        for key, value in black_log.items():
            log['Black_model   '+key] = value

        return log

    def shifting_attack(self, data_loader, threat_model, black_model):
        """[对一个模型进行噪声偏移的攻击]

        Args:
            data_loader ([DataLoader]): [数据加载器]
            threat_model ([Model]): [用于生成对抗样本的白盒模型]
            black_model ([Model]): [用于攻击的黑盒模型]

        Returns:
            acc [float]: [攻击后的准确率]
            mean [float]: 平均扰动大小
        """
        log = {}

        model_name = self.config.Black_Attack['threat_model']
        white_model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)
        white_log, real_sample, adv_sample, adv_pertub, targets, x_snrs = self.attack_set(white_model, data_loader)
        
        for key, value in white_log.items():
            log['White_model   '+key] = value


        torch.cuda.empty_cache()

        model_name = self.config.Black_Attack['black_model']
        black_model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)

        black_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample, targets, x_snrs)
        adv_loader = DataLoader(black_dataset, batch_size = 32, shuffle = False, num_workers = 4)

        black_log = self._model_test(black_model, adv_loader)


        for key, value in black_log.items():
            log['Black_model   '+key] = value

            
        torch.cuda.empty_cache()

        adv_sample_tranfer = adv_sample.copy()

        acc = 0
        snr_acc = np.zeros(len(self.snrs))
        for i in tqdm(range(128 - 1)):
            zero_shape = list(adv_pertub.shape)
            zero_shape[-1] = 1
            zero = np.zeros(tuple(zero_shape))
            adv_pertub_tranfer = np.concatenate((zero, adv_pertub),axis=-1)[:, :, :, :-1]
            adv_sample_tranfer = real_sample + adv_pertub_tranfer

            black_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample_tranfer, targets, x_snrs)
            adv_loader = DataLoader(black_dataset, batch_size = 32, shuffle = False, num_workers = 4)

            black_log = self._model_test(black_model, adv_loader)

            acc += black_log['accuracy']
            snr_acc += black_log['snr_acc']

    
        log['shifting_acc'] = acc/127
        log['shifting_snr_acc'] = snr_acc/127

        return log

    def _model_test(self, model, data_loader):
        predict = []
        targets = []
        snr_acc = np.zeros(len(self.snrs))
        snr_num = np.zeros(len(self.snrs))
        total_metrics = np.zeros(len(self.metrics))
        for idx,(x_adv, y, x_snr) in enumerate(data_loader):
            x_adv = torch.tensor(x_adv).to(self.device).float()
            y = np.array(y)

            logits = self.model(x_adv).cpu().detach().numpy()
            total_metrics += self._eval_metrics(logits, y)

            predict.append(logits)
            targets.append(y)

            logits = np.argmax(logits, axis=1)
            # print(logits)
            x_snr = x_snr.numpy()
            for i, snr in enumerate(self.snrs):
                if np.sum(x_snr== snr) != 0:
                    snr_acc[i] += np.sum(logits[x_snr == snr] == y[x_snr == snr])
                    snr_num[i] += np.sum(x_snr == snr)

        snr_acc = snr_acc / snr_num
        # self.plot_snr_figure(self.figure_dir, snr_acc, self.snrs)

        predict = np.vstack(predict)
        targets = np.hstack(targets)
        # self.plot_confusion_matrix_figure(self.figure_dir, predict, targets, self.mods)
        
        log = {}
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/len(data_loader)
        
        log['snr_acc'] = snr_acc

        return log


    def generate_data(self, data_loader):
        model_list = ['VTCNN2', 'Based_GRU', 'Based_VGG', 'Based_ResNet', 'CLDNN']
        # model_list = ['VTCNN2']
        data = []
        labels = []
        snrs = []
        print('开始数据处理')
        for idx,(x,y, x_snr) in enumerate(tqdm(data_loader)):
            data.append(x)
            labels.append(y)
            snrs.append(x_snr)
        print('已完成原始数据初始化')
        for model_name in model_list:
            print('开始模型{}校验'.format(model_name))
            model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)

            data, labels, snrs = self.judge_data(model, data, labels, snrs)
            print('已完成模型{}校验'.format(model_name))
            torch.cuda.empty_cache()
        data = np.vstack(data)
        labels = np.hstack(labels)
        snrs = np.hstack(snrs)
        dirname = '/home/yuzhen/wireless/RML2016.10a'
        print('-----还有{}数据'.format(len(data)))
        pickle.dump([data, labels, snrs], open(os.path.join(dirname, 'attack_data.p'), 'wb'))
        print('-----已完成数据存储-----')

    def judge_data(self, model, data, labels, snrs):
        fresh_data = []
        fresh_labels = []
        fresh_snrs = []
        for idx,(x, y, x_snr) in enumerate(tqdm(zip(data, labels, snrs))):
            x = torch.tensor(x).to(self.device).float()
            logits = model(x).cpu().detach().numpy()
            predict = np.argmax(logits, axis=1)
            x_snr = np.array(x_snr)
            y = np.array(y)
            index = (predict == y) & (x_snr >=0)
            if len(x[index] > 0):
                fresh_data.append(x[index].cpu().detach().numpy())
                fresh_labels.append(y[index])
                fresh_snrs.append(x_snr[index])
        return fresh_data, fresh_labels, fresh_snrs


    
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

    def _eval_metrics(self,logits,targets):
        """[多种metric的运算]

        Args:
            logits ([array]): [网络模型输出]
            targets ([array]): [标签值]

        Returns:
            acc_metrics [array]: [多个metric对应的结果]
        """
        acc_metrics = np.zeros(len(self.metrics))
        for i,metric in enumerate(self.metrics):
            acc_metrics[i] = metric(logits,targets)
        return acc_metrics
        