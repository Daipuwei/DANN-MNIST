# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 16:10
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : utils.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.core.framework import summary_pb2

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def plot_accuracy(x,y,path):
    """
    这是绘制精度的函数
    :param x: x坐标数组
    :param y: y坐标数组
    :param path: 结果保存地址
    :param mode: 模式，“train”代表训练损失，“val”为验证损失
    """
    lengend_array = ["train_acc", "val_acc"]
    train_accuracy,val_accuracy = y
    plt.plot(x, train_accuracy, 'r-')
    plt.plot(x, val_accuracy, 'b--')
    plt.grid(True)
    plt.xlim(0, x[-1]+2)
    #plt.xticks(x)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(lengend_array,loc="best")
    plt.savefig(path)
    plt.close()

def plot_loss(x,y,path,mode="train"):
    """
    这是绘制损失的函数
    :param x: x坐标数组
    :param y: y坐标数组
    :param path: 结果保存地址
    :param mode: 模式，“train”代表训练损失，“val”为验证损失
    """
    if mode == "train":
        lengend_array = ["train_loss","train_image_cls_loss","train_domain_cls_loss"]
    else:
        lengend_array = ["val_loss", "val_image_cls_loss", "val_domain_cls_loss"]
    loss_results,image_cls_loss_results,domain_cls_loss_results = y
    loss_results_min = np.max([np.min(loss_results) - 0.1,0])
    image_cls_loss_results_min = np.max([np.min(image_cls_loss_results) - 0.1,0])
    domain_cls_loss_results_min =np.max([np.min(domain_cls_loss_results) - 0.1,0])
    y_min = np.min([loss_results_min,image_cls_loss_results_min,domain_cls_loss_results_min])
    plt.plot(x, loss_results, 'r-')
    plt.plot(x, image_cls_loss_results, 'b--')
    plt.plot(x, domain_cls_loss_results, 'g-.')
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xlim(0,x[-1]+2)
    plt.ylim(ymin=y_min)
    #plt.xticks(x)
    plt.legend(lengend_array,loc="best")
    plt.savefig(path)
    plt.close()

def learning_rate_schedule(process,init_learning_rate = 0.01,alpha = 10.0 , beta = 0.75):
    """
    这个学习率的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param init_learning_rate: 初始学习率，默认为0.01
    :param alpha: 参数alpha，默认为10
    :param beta: 参数beta，默认为0.75
    """
    return init_learning_rate /(1.0 + alpha * process)**beta

def grl_lambda_schedule(process,gamma=10.0):
    """
    这是GRL的参数lambda的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param gamma: 参数gamma，默认为10
    """
    return 2.0 / (1.0+np.exp(-gamma*process)) - 1.0