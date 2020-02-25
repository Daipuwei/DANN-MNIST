# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 16:36
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : train.py
# @Software: PyCharm

import os
import numpy as np
import pickle as pkl

from config.config import config
from model.MNIST2MNIST_M import MNIST2MNIST_M_DANN
from tensorflow.examples.tutorials.mnist import input_data
from utils.utils import batch_generator

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    cfg = config()

    mnist = input_data.read_data_sets(os.path.abspath('./dataset/mnist'), one_hot=True)
    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    # Load MNIST-M
    mnistm = pkl.load(open(os.path.abspath('./dataset/mnistm/mnistm_data.pkl'), 'rb'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    mnistm_valid = mnistm['valid']

    # Compute pixel mean for normalizing data
    pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))
    cfg.set(pixel_mean = pixel_mean)

    # 构造数据生成器
    train_source_datagen = batch_generator([mnist_train,mnist.train.labels],cfg.batch_size // 2)
    train_target_datagen = batch_generator([mnistm_train,mnist.train.labels],cfg.batch_size // 2)
    val_datagen = batch_generator([mnistm_test,mnist.test.labels],cfg.batch_size)

    # 初始化每个epoch的训练次数和每次验证过程的验证次数
    train_source_batch_num = int(len(mnist_train) // (cfg.batch_size // 2))
    train_target_batch_num = int(len(mnistm_train) // (cfg.batch_size // 2))
    train_iter_num = int(np.max([train_source_batch_num,train_target_batch_num]))
    val_iter_num = int(len(mnistm_test) / cfg.batch_size)

    # 初始化相关参数
    interval = 2  # 验证间隔
    train_num = len(mnist_train) +  len(mnistm_train)# 训练集样本数
    val_num = len(mnistm_test)     # 验证集样本数
    print("train on %d training samples with batch_size %d ,validation on %d val samples"
          % (train_num, cfg.batch_size, val_num))

    # 初始化DANN，并进行训练
    dann = MNIST2MNIST_M_DANN(cfg)
    #pre_model_path = os.path.abspath("./pre_model/trained_model.ckpt")
    pre_model_path = None
    dann.train(train_source_datagen,train_target_datagen,val_datagen,pixel_mean,
               interval,train_iter_num,val_iter_num,pre_model_path)

if __name__ == '__main__':
    run_main()