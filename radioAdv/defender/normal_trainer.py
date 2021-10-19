from defender.base_trainer import BaseTrainer
import utils as util

import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import numpy as np


class NormalTrainer(BaseTrainer):
    """[LeNet+Mnist]
    """
    def __init__(self, model,train_loader, test_loader,criterion,optimizer,metrics,config, snrs = [], mods = []):
        
        super(NormalTrainer,self).__init__(model,train_loader, test_loader,criterion,optimizer,metrics,config)
        self.snrs = snrs
        self.mods = mods

    def _train_epoch(self,epoch):

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        trainloader_t = tqdm(self.train_loader,ncols=100)
        
        trainloader_t.set_description("Train Epoch: {}|{}  Batch size: {}  LR : {:.4}".format
                                      (epoch,self.EPOCH,self.train_loader.batch_size,self.optimizer.param_groups[0]['lr']))
        
        for idx,(x, y, x_snr) in enumerate(trainloader_t):
            if self.use_gpu:
                x = x.to(self.device)
                y = y.to(self.device)
            x = x.float()
            y = y.long()
            self.optimizer.zero_grad()

            logits = self.model(x)

            loss = self.criterion(logits,y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(logits.cpu().detach().numpy(),y.cpu().detach().numpy())
        
        log = {
            'loss' : total_loss /self.len_epoch,
        }
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/self.len_epoch

        return log


    def _test_epoch(self):
        total_metrics = np.zeros(len(self.metrics))

        test_loader_t = tqdm(self.test_loader,ncols=100)
        test_loader_t.set_description("Batch size: {}".format(self.test_loader.batch_size))

        predict = []
        targets = []
        snr_acc = np.zeros(len(self.snrs))
        snr_num = np.zeros(len(self.snrs))
        for idx,(x, y, x_snr) in enumerate(test_loader_t):
            if self.use_gpu:
                x = x.to(self.device)
                y = y.to(self.device)
            x = x.float()
            y = y.long().cpu().detach().numpy()
            logits = self.model(x).cpu().detach().numpy()
            total_metrics += self._eval_metrics(logits,y)

            predict.append(logits)
            targets.append(y)

            logits = np.argmax(logits, axis=1)
            x_snr = x_snr.numpy()
            for i, snr in enumerate(self.snrs):
                if np.sum(x_snr== snr) != 0:
                    snr_acc[i] += np.sum(logits[x_snr == snr] == y[x_snr == snr])
                    snr_num[i] += np.sum(x_snr == snr)

        snr_acc = snr_acc / snr_num
        self.plot_snr_figure(self.figure_dir, snr_acc, self.snrs)

        predict = np.vstack(predict)
        targets = np.hstack(targets)
        self.plot_confusion_matrix_figure(self.figure_dir, predict, targets, self.mods)
        
        log = {}
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/self.len_epoch
        
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
        plt.switch_backend('agg')
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%m-%d_%H-%M")

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
        plt.switch_backend('agg')
        cm = util.generate_confusion_matrix(predict, targets, mods)
        util.ensure_dir(dirname)
        util.plot_confusion_matrix(cm, dirname, mods)
        print("Figure 'Confusion Matrix' generated successfully")

