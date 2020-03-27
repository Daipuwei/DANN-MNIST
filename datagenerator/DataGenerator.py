# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 12:19
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : DataGenerator.py
# @Software: PyCharm

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

class DataGenerator(object):

    def __init__(self,dataset_dir,batch_size,size=28,source_flag=True,mode="train"):
        """
        这是MNINST2MNINST_M数据集的批量数据生成类的初始化函数
        :param dataset_dir: 数据集目录
        :param batch_size: 小批量规模
        :param mode: 模式，“train”为读取训练集，“val”则是读取测试集
        """
        # 初始化相关参数
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.mode = mode
        self.source_flag = source_flag
        self.size = size

        self.shuffle_flag = True            # mnist数据集是否随机打乱随机打乱标志
        self.start = 0
        self.end = 0

        #  根据不同模式获取mnist与mnistM数据集目录
        #self.dataset_dir = dataset_dir
        if mode == "train":
            self.dataset_dir = os.path.join(self.dataset_dir,"train")
        else:
            self.dataset_dir = os.path.join(self.dataset_dir,"test")

        # 获取mode模式下的mnist与mnistM数据集的图像地址及其标签
        self.dataset_paths, self.labels = self.get_image_label_dataset()

        """
        # 在程序编写后，用少量数据集验证程序是否出现bug，节省时间
        self.dataset_paths = self.dataset_paths[:128]
        self.labels = self.labels[:128,:]
        """

        # 最后一个小批量数据集规模可能小于batch_size ,舍弃用，更新对应数据集规模
        self.batch_num = int(len(self.dataset_paths) / self.batch_size)
        self.dataset_size = self.batch_num * self.batch_size

    def get_image_label_dataset(self):
        """
        这是获取图像地址和标签的函数
        :return:
        """
        dataset_paths = []
        labels = []
        for label in os.listdir(self.dataset_dir):
            image_dir = os.path.join(self.dataset_dir,label)
            label_image_paths = list([os.path.join(self.dataset_dir,label,image_name)
                                       for image_name in os.listdir(image_dir)])
            dataset_paths.extend(label_image_paths)
            labels.extend([int(label)]*len(label_image_paths))
        return np.array(dataset_paths),to_categorical(np.array(labels))

    def get_batch_image_label(self,batch_image_paths):
        """
        这是读取批量数据集与域标签的函数
        :param batch_image_paths: 批量数据集
        :param mode: 模式，“mnist”则生成全0的域标签，“mnist_m”则生成全1的域标签
        :return:
        """
        size = len(batch_image_paths)
        if self.source_flag == True:        # 源域数据，域标签置为0
            batch_domain_labels = np.tile([1., 0.], [size, 1])
        else:                               # 目标域数据，域标签置为1
            batch_domain_labels = np.tile([0., 1.], [size, 1])
        batch_image_data = []
        for image_path in batch_image_paths:
            img = cv2.resize(cv2.imread(image_path),(self.size,self.size))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            """
            # 50%的概率水平翻转图像
            if np.random.rand() > 0.5:
                img = cv2.flip(img,1)
            """
            batch_image_data.append(img)
        return np.array(batch_image_data),batch_domain_labels

    def data_generator(self):
        """
        这是生成批量图像数据生成器的函数
        :return:
        """
        self.start = 0
        while True:
            if self.shuffle_flag == True:            # 随机打乱数据集
                random_index = np.random.permutation(len(self.labels))
                self.dataset_paths = self.dataset_paths[random_index]
                self.labels = self.labels[random_index]
                self.shuffle_flag = False
                self.start = 0
                self.end = 0

            # 读取小批量数据集
            self.end = self.start + self.batch_size
            batch_image_paths = self.dataset_paths[self.start:self.end]
            batch_labels = self.labels[self.start:self.end,:]
            batch_image_data,batch_domain_labels = self.get_batch_image_label(batch_image_paths)
            if self.end == self.dataset_size:          # 遍历完数据集，将随机打乱标志置为True
                self.mninst_shuffle_flag = True
            self.start = self.end

            yield batch_image_data,batch_labels,batch_domain_labels

    def next_batch(self):
        return self.data_generator().__next__()