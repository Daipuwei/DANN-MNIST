# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 上午1:43
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : train_MNIST2MNIST_M.py
# @Software: PyCharm


import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from config.config import config
from utils.dataset_utils import batch_generator
from model.MNIST2MNIST_M_train import MNIST2MNIST_M_DANN

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    image_input_shape = (28, 28, 3)
    image_size = 28
    init_learning_rate = 1e-2
    momentum_rate = 0.9
    batch_size = 256
    epoch = 5000
    pre_model_path = None
    checkpoints_dir = os.path.abspath("./checkpoints/MNIST2MNIST_M/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size))
    logs_dir = os.path.abspath("./logs/MNIST2MNIST_M/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size))
    config_dir = os.path.abspath("./config/MNIST2MNIST_M/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size))
    source_dataset_path = os.path.abspath("./model_data/dataset/MNIST/mnist_data.pkl")
    target_dataset_path = os.path.abspath("./model_data/dataset/MNIST_M/mnist_m_data.pkl")
    cfg = config(pre_model_path = pre_model_path,
                 checkpoints_dir = checkpoints_dir,
                 logs_dir = logs_dir,
                 config_dir = config_dir,
                 source_dataset_path = source_dataset_path,
                 target_dataset_path = target_dataset_path,
                 image_input_shape = image_input_shape,
                 image_size = image_size,
                 init_learning_rate = init_learning_rate,
                 momentum_rate = momentum_rate,
                 batch_size = batch_size,
                 epoch = epoch)

    # Load MNIST
    mnist = pkl.load(open(os.path.abspath('model_data/dataset/MNIST/mnist_data.pkl'), 'rb'))
    mnist_x_train = mnist['train'].astype(np.float32)
    mnist_x_val = mnist['val'].astype(np.float32)
    mnist_y_train = mnist['train_label'].astype(np.int32)
    mnist_y_val = mnist['val_label'].astype(np.int32)
    mnist_y_train = to_categorical(mnist_y_train).astype(np.float32)
    mnist_y_val= to_categorical(mnist_y_val).astype(np.float32)

    # Load MNIST-M
    mnistm = pkl.load(open(os.path.abspath('./model_data/dataset/MNIST_M/mnist_m_data.pkl'), 'rb'))
    mnistm_x_train = mnistm['train'].astype(np.float32)
    mnistm_x_val = mnistm['val'].astype(np.float32)

    # Compute pixel mean for normalizing data
    pixel_mean = np.vstack([mnist_x_train, mnistm_x_train]).mean((0, 1, 2))
    cfg.set(pixel_mean=pixel_mean)
    print(pixel_mean)

    mnist_x_train = (mnist_x_train - pixel_mean) / 255.0
    mnistm_x_train = (mnistm_x_train - pixel_mean) / 255.0
    mnist_x_val = (mnist_x_val - pixel_mean) / 255.0
    mnistm_val = (mnistm_x_val - pixel_mean) / 255.0

    # 构造数据生成器
    train_source_datagen = batch_generator([mnist_x_train,mnist_y_train],cfg.batch_size // 2)
    train_target_datagen = batch_generator([mnistm_x_train,mnist_y_train],cfg.batch_size // 2)
    val_target_datagen = batch_generator([mnistm_val,mnist_y_val],cfg.batch_size)

    # 初始化每个epoch的训练次数和每次验证过程的验证次数
    train_source_batch_num = int(len(mnist_x_train) // (cfg.batch_size // 2))
    train_target_batch_num = int(len(mnistm_x_train) // (cfg.batch_size // 2))
    train_iter_num = int(np.max([train_source_batch_num,train_target_batch_num]))
    val_iter_num = int(len(mnistm_x_val) // cfg.batch_size)

    # 初始化DANN，并进行训练
    dann = MNIST2MNIST_M_DANN(cfg)
    dann.train(train_source_datagen,train_target_datagen,val_target_datagen,train_iter_num,val_iter_num)

if __name__ == '__main__':
    run_main()
