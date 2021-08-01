# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 下午1:50
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : create_mnistm.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
import pickle as pkl
import skimage.io
import skimage.transform
from tensorflow.keras.datasets import mnist

rand = np.random.RandomState(42)

def compose_image(mnist_data, background_data):
    """
    这是将MNIST数据和BSDS500数据进行融合成MNIST-M数据的函数
    :param mnist_data: MNIST数据
    :param background_data: BDSD500数据，作为背景图像
    :return:
    """
    # 随机融合MNIST数据和BSDS500数据
    w, h, _ = background_data.shape
    dw, dh, _ = mnist_data.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background_data[x:x + dw, y:y + dh]
    return np.abs(bg - mnist_data).astype(np.uint8)

def mnist_to_img(x):
    """
    这是实现MNIST数据格式转换的函数，0/1数据位转化为RGB数据集
    :param x: 0/1格式MNIST数据
    :return:
    """
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)

def create_mnistm(X,background_data):
    """
    这是生成MNIST-M数据集的函数，MNIST-M数据集介绍可见：
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    :param X: MNIST数据集
    :param background_data: BSDS500数据集，作为背景
    :return:
    """
    # 遍历所有MNIST数据集，生成MNIST-M数据集
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        if i % 1000 == 0:
            print('Processing example', i)
        # 随机选择背景图像
        bg_img = rand.choice(background_data)
        # 0/1数据位格式MNIST数据转换为RGB格式
        mnist_image = mnist_to_img(X[i])
        # 将MNIST数据和BSDS500数据背景进行融合
        mnist_image = compose_image(mnist_image, bg_img)
        X_[i] = mnist_image
    return X_

def run_main():
    """
    这是主函数
    """
    # 初始化路径
    BST_PATH = os.path.abspath('./model_data/dataset/BSR_bsds500.tgz')
    mnist_dir = os.path.abspath("model_data/dataset/MNIST")
    mnistm_dir = os.path.abspath("model_data/dataset/MNIST_M")

    # 导入MNIST数据集
    (X_train,y_train),(X_test,y_test) = mnist.load_data()


    # 加载BSDS500数据集
    f = tarfile.open(BST_PATH)
    train_files = []
    for name in f.getnames():
        if name.startswith('BSR/BSDS500/data/images/train/'):
            train_files.append(name)
    print('Loading BSR training images')
    background_data = []
    for name in train_files:
        try:
            fp = f.extractfile(name)
            bg_img = skimage.io.imread(fp)
            background_data.append(bg_img)
        except:
            continue

    # 生成MNIST-M训练数据集和验证数据集
    print('Building train set...')
    train = create_mnistm(X_train,background_data)
    print(np.shape(train))
    print('Building validation set...')
    valid = create_mnistm(X_test,background_data)
    print(np.shape(valid))

    # 将MNIST数据集转化为RGB格式
    X_train = np.expand_dims(X_train,-1)
    X_test = np.expand_dims(X_test,-1)
    X_train = np.concatenate([X_train,X_train,X_train],axis=3)
    X_test = np.concatenate([X_test,X_test,X_test],axis=3)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)
    # 保存MNIST数据集为pkl文件
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir)
    with open(os.path.join(mnist_dir, 'mnist_data.pkl'), 'wb') as f:
        pkl.dump({'train': X_train,
                  'train_label': y_train,
                  'val': X_test,
                  'val_label':y_test}, f, pkl.HIGHEST_PROTOCOL)

    # 保存MNIST-M数据集为pkl文件
    if not os.path.exists(mnistm_dir):
        os.mkdir(mnistm_dir)
    with open(os.path.join(mnistm_dir, 'mnist_m_data.pkl'), 'wb') as f:
        pkl.dump({'train': train,
                  'train_label':y_train,
                  'val': valid,
                  'val_label':y_test}, f, pkl.HIGHEST_PROTOCOL)

    # 计算数据集平均值，用于数据标准化
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(train))
    print(np.shape(valid))
    print(np.shape(y_train))
    print(np.shape(y_test))
    pixel_mean = np.vstack([X_train,train,X_test,valid]).mean((0,1,2))
    print(np.shape(pixel_mean))
    print(pixel_mean)

if __name__ == '__main__':
    run_main()
