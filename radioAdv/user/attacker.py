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



    def attack_batch(self, model, x, y, x_snr):
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
        # x = torch.Tensor(x.float()).to(self.device)
        # # print(x[0])
        # y = torch.Tensor(y.float()).to(self.device).long()
        x = x.to(self.device).float()
        y = y.to(self.device).long()

        x_advs, pertubations, logits, nowLabels = self.attack_method.attack(
            x, y, x_snr, **getattr(self.config, self.config.CONFIG['attack_name']))

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
        x_adv,pertubation,nowLabel = self.attack_batch(self.model, x, y, None)
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
        now_labels = []

        real_sample = []
        adv_sample = []
        adv_pertub = []
        x_snrs = []

        snr_acc = np.zeros(len(self.snrs))
        snr_num = np.zeros(len(self.snrs))

        for idx,(x,y, x_snr) in enumerate(tqdm(data_loader)):
            x_advs ,pertubations, logits, nowLabels = self.attack_batch(model, x, y, x_snr)
            y = y.detach().cpu().numpy()
            # 计算平均攻击成功率, 
            if self.is_target:
                success_num += ((self.target == nowLabels) == (self.target != y)).sum()
                data_num += (self.target != y).sum()
            else:
                data_num +=  x.shape[0]
                success_num += (y != nowLabels).sum()
                # print((y == nowLabels).sum())
                # print(y)
                # print(nowLabels)

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
            now_labels.append(nowLabels)

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

        predict = np.hstack(now_labels)
        targets = np.hstack(targets)

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
            log = self.shifting_attack(data_loader, threat_model, black_model, **getattr(self.config, attack_method))

        elif attack_method == 'White_Attack_Average':
            log = self.white_attack_average(data_loader)
        elif attack_method == 'Black_Attack_Average':
            threat_model = self.config.Black_Attack['threat_model']
            black_model = self.config.Black_Attack['black_model']
            log = self.black_attack_average(data_loader, threat_model, black_model)
        elif attack_method == 'Shifting_Attack_Average':
            threat_model = self.config.Black_Attack['threat_model']
            black_model = self.config.Black_Attack['black_model']
            log = self.shifting_attack_average(data_loader, threat_model, black_model, **getattr(self.config, 'Shifting_Attack'))


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

        # 对Shifting_Noise_Extend攻击方式的噪声进行特殊处理，对每一个扰动，都进行随机的选择
        if self.config.CONFIG['attack_name'] == 'Shifting_Noise_Extend':
            pertubation = []
            shift = self.config.Shifting_Noise_Extend['shift']
            for i in range(adv_pertub.shape[0]):
                random_shift = np.random.randint(0, 2*shift + 1)
                random_pertubation = adv_pertub[i, :, :, random_shift : random_shift + 128]
                pertubation.append(random_pertubation)
            pertubation = np.stack(pertubation)
            adv_sample = real_sample + pertubation


        black_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample, targets, x_snrs)
        adv_loader = DataLoader(black_dataset, batch_size = 32, shuffle = False, num_workers = 4)

        black_log = self._model_test(black_model, adv_loader)


        for key, value in black_log.items():
            log['Black_model   '+key] = value

        return log

    def shifting_attack(self, data_loader, threat_model, black_model, load_parameter, parameter_path, shift_k):
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


        #  进行白盒攻击， 得到对抗样本
        model_name = self.config.Black_Attack['threat_model']
        white_model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)
        if not load_parameter:
            white_log, real_sample, adv_sample, adv_pertub, targets, x_snrs = self.attack_set(white_model, data_loader)
        else:
            real_sample, adv_sample, targets, x_snrs, adv_pertub = self.load_pertub_parameter(parameter_path)
            shift_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample, targets, x_snrs)
            shift_loader = DataLoader(shift_dataset, batch_size = 32, shuffle = False, num_workers = 4)
            white_log = self._model_test(white_model, shift_loader)

        pertub_mean = np.mean(np.abs(adv_pertub))
        log['pertub_mean'] = pertub_mean

        for key, value in white_log.items():
            log['White_model   '+key] = value


        torch.cuda.empty_cache()

        #  进行黑盒攻击
        model_name = self.config.Black_Attack['black_model']
        black_model = getattr(model_loader, 'load' + model_name)(**getattr(self.config, model_name)).to(self.device)

        # 对Shifting_Noise_Extend攻击方式的噪声进行特殊处理，对每一个扰动，都进行随机的选择
        if self.config.CONFIG['attack_name'] == 'Shifting_Noise_Extend':
            pertubation = []
            shift = self.config.Shifting_Noise_Extend['shift']
            for i in range(adv_pertub.shape[0]):
                random_shift = np.random.randint(0, 2*shift + 1)
                random_pertubation = adv_pertub[i, :, :, random_shift : random_shift + 128]
                pertubation.append(random_pertubation)
            pertubation = np.stack(pertubation)
            adv_sample = real_sample + pertubation
            
        black_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample, targets, x_snrs)
        adv_loader = DataLoader(black_dataset, batch_size = 32, shuffle = False, num_workers = 4)

        black_log = self._model_test(black_model, adv_loader)


        for key, value in black_log.items():
            log['Black_model   '+key] = value

            
        torch.cuda.empty_cache()

        # 进行shifting 黑盒攻击
        black_model = black_model.to(self.device)
        adv_sample_tranfer = adv_sample.copy()
        

        acc = 0
        snr_acc = np.zeros(len(self.snrs))
        shift_acc_list = []
        for k in tqdm(range(-shift_k, shift_k+1, 1)):
            # 对Shifting_Noise_Extend攻击方式的噪声进行特殊处理，对每一个扰动，都进行随机的选择
            if self.config.CONFIG['attack_name'] == 'Shifting_Noise_Extend':
                shift = self.config.Shifting_Noise_Extend['shift']

                first_pertubation = []          # 第一次发射的扰动
                second_pertubation = []         # 第二次发射的扰动
                for i in range(adv_pertub.shape[0]):
                    random_shift = np.random.randint(0, 2*shift + 1)
                    random_pertubation = adv_pertub[i, :, :, random_shift : random_shift + 128]
                    first_pertubation.append(random_pertubation)

                    random_shift = np.random.randint(0, 2*shift + 1)
                    random_pertubation = adv_pertub[i, :, :, random_shift : random_shift + 128]
                    second_pertubation.append(random_pertubation)
                first_pertubation = np.stack(pertubation)
                second_pertubation = np.stack(pertubation)
            
            else:
                first_pertubation = adv_pertub.copy()
                second_pertubation = adv_pertub.copy()

            if k > 0:
                # 扰动在信号的后面
                first_noise = first_pertubation[:, :, :, 128 - k: 128]
                second_noise = second_pertubation[:, :, :, : 128 - k]
                adv_pertub_tranfer = np.concatenate((first_noise, second_noise),axis=-1)
            elif k < 0:
                # 扰动在信号的前面
                first_noise = first_pertubation[:, :, :, -k : 128]
                second_noise = second_pertubation[:, :, :, : -k]
                adv_pertub_tranfer = np.concatenate((first_noise, second_noise),axis=-1)
            else:
                adv_pertub_tranfer = first_pertubation

            adv_sample_tranfer = real_sample + adv_pertub_tranfer

            black_dataset = module_data_loader.Rml2016_10aAdvSampleSet(adv_sample_tranfer, targets, x_snrs)
            adv_loader = DataLoader(black_dataset, batch_size = 32, shuffle = False, num_workers = 4)

            black_log = self._model_test(black_model, adv_loader)

            acc += black_log['accuracy']
            snr_acc += black_log['snr_acc']
            shift_acc_list.append(black_log['accuracy'])

    
        log['shifting_acc'] = acc/(2 * shift_k + 1)
        log['shifting_snr_acc'] = snr_acc/(2 * shift_k + 1)
        log['shift_acc_list'] = shift_acc_list

        return log

    def white_attack_average(self, data_loader):
        # 该函数仅针对Shifting_Noise_Extend,由于其随机性，计算其平均成功率
        log = {}
        acc_average = 0
        pertubmean_average = 0
        pertub_prop_average = 0
        pertub_max_average = 0
        snr_acc_average = np.zeros(10)
        for i in range(10):
            print('第{}次白盒攻击，开始--------------'.format(i))
            attack_log = self.white_attack(data_loader)

            acc_average += attack_log['acc']
            pertubmean_average += attack_log['pertubmean']
            pertub_prop_average += attack_log['pertub_prop']
            pertub_max_average += attack_log['pertub_max']
            snr_acc_average += np.array(attack_log['snr_acc'])

        log['acc_average'] = acc_average/10
        log['pertubmean_average'] = pertubmean_average/10
        log['pertub_prop_average'] = pertub_prop_average/10
        log['pertub_max_average'] = pertub_max_average/10
        log['snr_acc_average'] = snr_acc_average/10
        return log

    def black_attack_average(self, data_loader, threat_model, black_model):
        # 该函数仅针对Shifting_Noise_Extend,由于其随机性，计算其平均成功率
        log = {}
        white_acc_average = 0
        white_pertubmean_average = 0
        white_pertub_prop_average = 0
        white_pertub_max_average = 0
        white_snr_acc_average = np.zeros(10)

        black_acc_average = 0
        black_snr_acc_average = np.zeros(10)

        for i in range(10):
            print('第{}次黑盒攻击，开始--------------'.format(i))
            attack_log = self.black_attack(data_loader, threat_model, black_model)
            
            white_acc_average += attack_log['White_model   acc']
            white_pertubmean_average += attack_log['White_model   pertubmean']
            white_pertub_prop_average += attack_log['White_model   pertub_prop']
            white_pertub_max_average += attack_log['White_model   pertub_max']
            white_snr_acc_average += np.array(attack_log['White_model   snr_acc'])

            black_acc_average += attack_log['Black_model   accuracy']
            black_snr_acc_average += np.array(attack_log['Black_model   snr_acc'])


        log['White_model   acc_average'] = white_acc_average/10
        log['White_model   pertubmean_average'] = white_pertubmean_average/10
        log['White_model   pertub_prop_average'] = white_pertub_prop_average/10
        log['White_model   pertub_max_average'] = white_pertub_max_average/10
        log['White_model   snr_acc_average'] = white_pertub_max_average/10

        log['Black_model   acc_average'] = black_acc_average/10
        log['Black_model   snr_acc_average'] = black_snr_acc_average/10
        return log

    def shifting_attack_average(self, data_loader, threat_model, black_model, load_parameter, parameter_path, shift_k):
        # 该函数仅针对Shifting_Noise_Extend,由于其随机性，计算其平均成功率
        log = {}
        white_acc_average = 0
        white_pertubmean_average = 0
        white_pertub_prop_average = 0
        white_pertub_max_average = 0
        white_snr_acc_average = np.zeros(10)

        black_acc_average = 0
        black_snr_acc_average = np.zeros(10)

        shift_acc_average = 0
        shift_snr_acc_average = np.zeros(10)
        for i in range(10):
            print('第{}次平移攻击，开始--------------'.format(i))
            attack_log = self.shifting_attack(data_loader, threat_model, black_model, load_parameter, parameter_path, shift_k)
            
            white_acc_average += attack_log['White_model   acc']
            white_pertubmean_average += attack_log['White_model   pertubmean']
            white_pertub_prop_average += attack_log['White_model   pertub_prop']
            white_pertub_max_average += attack_log['White_model   pertub_max']
            white_snr_acc_average += np.array(attack_log['White_model   snr_acc'])

            black_acc_average += attack_log['Black_model   accuracy']
            black_snr_acc_average += np.array(attack_log['Black_model   snr_acc'])

            shift_acc_average += attack_log['shifting_acc']
            shift_snr_acc_average += np.array(attack_log['shifting_snr_acc'])


        log['White_model   acc_average'] = white_acc_average/10
        log['White_model   pertubmean_average'] = white_pertubmean_average/10
        log['White_model   pertub_prop_average'] = white_pertub_prop_average/10
        log['White_model   pertub_max_average'] = white_pertub_max_average/10
        log['White_model   snr_acc_average'] = white_snr_acc_average/10

        log['Black_model   acc_average'] = black_acc_average/10
        log['Black_model   snr_acc_average'] = black_snr_acc_average/10

        log['Shifting_model   acc_average'] = shift_acc_average/10
        log['Shifting_model   snr_acc_average'] = shift_snr_acc_average/10
        return log
        

    def load_pertub_parameter(self, parameter_path):
        if not os.path.exists(parameter_path):
            print('there is no '+ parameter_path)
            return 
        x_list, y_list, x_snr_list, pretubation_mean_list, pretubation_list_list = pickle.load(open(parameter_path, 'rb'))
        data = np.vstack(np.array(x_list))
        targets = np.hstack(np.array(y_list))
        x_snrs = np.hstack(np.array(x_snr_list))


        adv_sample = []
        adv_pertub = []
        # print(len(x_list))
        for i in range(len(x_list)):
            pertubation_list = pretubation_list_list[i]
            zero_shape = list(pertubation_list.shape)
            
            pertubation_shape = zero_shape.copy()
            pertubation_shape[1] = 1
            pertubation_after_shift = np.zeros(tuple(pertubation_shape))

            zero_shape[1] = 1
            zero_shape[-1] = 1
            zero = np.zeros(tuple(zero_shape))

            
            for shift in range(128):
                pertubation_shift = pertubation_list[:, shift:shift+1, :, :]
                adv_pertub_tranfer = np.concatenate((pertubation_shift, zero),axis=-1)[:, :, :, 1:]
                pertubation_after_shift += adv_pertub_tranfer

            # divide_ratio = np.array([i+1 for i in range(128)])

            # pretubation_mean = pertubation_after_shift/divide_ratio

            # multi_ratio = np.array([i/100 for i in range(128, 0, -1)])
            # multi_ratio = np.exp(multi_ratio)
            # multi_ratio = multi_ratio / np.sum(multi_ratio)

            # pertubation_after_shift = pertubation_after_shift * multi_ratio
            x_adv = x_list[i] + pertubation_after_shift

            adv_sample.append(x_adv)
            adv_pertub.append(pertubation_after_shift)

        adv_sample = np.vstack(adv_sample)
        adv_pertub = np.vstack(adv_pertub)

        return data, adv_sample, targets, x_snrs, adv_pertub

            # print(pretubation_list.shape, x_list[i].shape)


    def _model_test(self, model, data_loader):
        predict = []
        targets = []
        snr_acc = np.zeros(len(self.snrs))
        snr_num = np.zeros(len(self.snrs))
        total_metrics = np.zeros(len(self.metrics))
        for idx,(x_adv, y, x_snr) in enumerate(data_loader):
            x_adv = torch.tensor(x_adv).to(self.device).float()
            y = np.array(y)

            logits = model(x_adv).cpu().detach().numpy()
            

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
        total_metrics += self._eval_metrics(predict, targets)
        # self.plot_confusion_matrix_figure(self.figure_dir, predict, targets, self.mods)
        
        log = {}
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]
        
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
        dirname = '/home/baiding/Study/research/radio'
        print(dirname)
        print('-----还有{}数据'.format(len(data)))
        pickle.dump([data, labels, snrs], open(os.path.join(dirname, 'attack_data.p'), 'wb'))
        print('-----已完成数据存储-----')

    def pick_data(self, data_loader):
        """[从数据集中挑选部分数据集]

        Args:
            data_loader ([type]): [description]
        """
        sample_matrix = np.ones((11, 10)) * 20
        data = []
        labels = []
        snrs = []
        for idx,(x,y, x_snr) in enumerate(tqdm(data_loader)):
            x = np.array(x)
            y = np.array(y)
            x_snr = np.array(x_snr)
            index = x_snr>= 0
            after_snr_x = x[index]
            after_snr_y = y[index]
            after_snr_snr = x_snr[index]
            for (one_x, one_y, one_x_snr) in zip(after_snr_x, after_snr_y, after_snr_snr):
                if sample_matrix[one_y, one_x_snr//2] > 0:
                    sample_matrix[one_y, one_x_snr//2] -= 1
                    data.append(one_x)
                    labels.append(one_y)
                    snrs.append(one_x_snr)
            if np.max(sample_matrix) == 0:
                break
        data = np.stack(data)
        labels = np.array(labels)
        snrs = np.array(snrs)
        print('-----生成了{}个数据'.format(len(data)))
        dirname = '/home/baiding/Study/research/radio'
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
        