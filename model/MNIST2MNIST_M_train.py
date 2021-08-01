# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 下午12:55
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : MNIST2MNIST_M_train.py
# @Software: PyCharm

import os
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy

from model.MNIST2MNIST_M import build_feature_extractor
from model.MNIST2MNIST_M import build_image_classify_extractor
from model.MNIST2MNIST_M import build_domain_classify_extractor
from model.GradientReveresalLayer import GradientReversalLayer

from utils.model_utils import EarlyStopping
from utils.model_utils import grl_lambda_schedule
from utils.model_utils import learning_rate_schedule

class MNIST2MNIST_M_DANN(object):

    def __init__(self,config):
        """
        这是MNINST与MNIST_M域适配网络的初始化函数
        :param config: 参数配置类
        """
        # 初始化参数类
        self.cfg = config

        # 定义相关占位符
        self.grl_lambd = 1.0              # GRL层参数

        # 搭建深度域适配网络
        self.build_DANN()

        # 定义训练和验证损失与指标
        self.loss = categorical_crossentropy
        self.acc = categorical_accuracy

        self.train_loss = Mean("train_loss", dtype=tf.float32)
        self.train_image_cls_loss = Mean("train_image_cls_loss", dtype=tf.float32)
        self.train_domain_cls_loss = Mean("train_domain_cls_loss", dtype=tf.float32)
        self.train_image_cls_acc = Mean("train_image_cls_acc", dtype=tf.float32)
        self.train_domain_cls_acc = Mean("train_domain_cls_acc", dtype=tf.float32)
        self.val_loss = Mean("val_loss", dtype=tf.float32)
        self.val_image_cls_loss = Mean("val_image_cls_loss", dtype=tf.float32)
        self.val_domain_cls_loss = Mean("val_domain_cls_loss", dtype=tf.float32)
        self.val_image_cls_acc = Mean("val_image_cls_acc", dtype=tf.float32)
        self.val_domain_cls_acc = Mean("val_domain_cls_acc", dtype=tf.float32)

        # 定义优化器
        self.optimizer = tf.keras.optimizers.SGD(self.cfg.init_learning_rate,
                                                 momentum=self.cfg.momentum_rate)

        '''
        # 初始化早停策略
        self.early_stopping = EarlyStopping(min_delta=1e-5, patience=100, verbose=1)
        '''

    def build_DANN(self):
        """
        这是搭建域适配网络的函数
        :return:
        """
        # 定义源域、目标域的图像输入和DANN模型图像输入
        self.image_input = Input(shape=self.cfg.image_input_shape,name="image_input")

        # 域分类器与图像分类器的共享特征
        self.feature_encoder = build_feature_extractor()
        # 获取图像分类结果和域分类结果张量
        self.image_cls_encoder = build_image_classify_extractor()
        self.domain_cls_encoder = build_domain_classify_extractor()

        self.grl = GradientReversalLayer()

        self.dann_model = Model(self.image_input,
                                [self.image_cls_encoder(self.feature_encoder(self.image_input)),
                                 self.domain_cls_encoder(self.grl(self.feature_encoder(self.image_input)))])
        self.dann_model.summary()

        # 导入
        if self.cfg.pre_model_path is not None:
            self.dann_model.load_weights(self.cfg.pre_model_path,by_name=True,skip_mismatch=True)

    def train(self,train_source_datagen,train_target_datagen,
              val_target_datagen,train_iter_num,val_iter_num):
        """
        这是DANN的训练函数
        :param train_source_datagen: 源域训练数据集生成器
        :param train_target_datagen: 目标域训练数据集生成器
        :param val_datagen: 验证数据集生成器
        :param train_iter_num: 每个epoch的训练次数
        :param val_iter_num: 每次验证过程的验证次数
        """
        # 初始化相关文件目录路径
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(self.cfg.checkpoints_dir,time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        log_dir = os.path.join(self.cfg.logs_dir, time)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.cfg.save_config(time)

        self.writer_hyperparameter = tf.summary.create_file_writer(os.path.join(log_dir,"hyperparameter"))
        self.writer_train = tf.summary.create_file_writer(os.path.join(log_dir,"train"))
        self.writer_val = tf.summary.create_file_writer(os.path.join(log_dir,'validation'))

        print('\n----------- start to train -----------\n')
        with open(os.path.join(log_dir,'log.txt'),'w') as f:
            for ep in np.arange(1,self.cfg.epoch+1,1):
                # 初始化精度条
                self.progbar = Progbar(train_iter_num+1)
                print('Epoch {}/{}'.format(ep, self.cfg.epoch))

                # 进行一个周期的模型训练
                train_loss,train_image_cls_acc = self.train_one_epoch\
                    (train_source_datagen,train_target_datagen,train_iter_num,ep)
                # 进行一个周期的模型验证
                val_loss,val_image_cls_acc = self.eval_one_epoch(val_target_datagen,val_iter_num,ep)
                # 更新进度条
                self.progbar.update(train_iter_num+1, [('val_loss', val_loss),
                                                       ("val_image_acc", val_image_cls_acc)])
                # 损失和指标清零
                self.train_loss.reset_states()
                self.train_image_cls_acc.reset_states()
                self.train_domain_cls_loss.reset_states()
                self.train_image_cls_acc.reset_states()
                self.train_domain_cls_acc.reset_states()
                self.val_loss.reset_states()
                self.val_image_cls_acc.reset_states()
                self.val_domain_cls_loss.reset_states()
                self.val_image_cls_acc.reset_states()
                self.val_domain_cls_acc.reset_states()

                # 保存训练过程中的模型
                str = "Epoch{:03d}-train_loss-{:.3f}-val_loss-{:.3f}-train_imgae_cls_acc-{:.3f}-val_image_cls_acc-{:.3f}"\
                    .format(ep, train_loss, val_loss,train_image_cls_acc,val_image_cls_acc)
                print(str)
                f.write(str+"\n")           # 写入日志文件
                self.dann_model.save(os.path.join(checkpoint_dir, str + ".h5"))

                '''
                # 判断是否需要早停模型训练过程，判断指标为目标域的图像分类精度
                stop_training = self.early_stopping.on_epoch_end(ep, val_image_cls_acc)
                if stop_training:
                    break
                '''
        self.dann_model.save(os.path.join(checkpoint_dir, "trained_dann_mnist2mnist_m.h5"))
        print('\n----------- end to train -----------\n')

    def train_one_epoch(self,train_source_datagen,train_target_datagen,train_iter_num,ep):
        """
        这是一个周期模型训练的函数
        :param train_source_datagen: 源域训练数据集生成器
        :param train_target_datagen: 目标域训练数据集生成器
        :param train_iter_num: 一个训练周期的迭代次数
        :param ep: 当前训练周期
        :return:
        """
        for i in np.arange(1, train_iter_num + 1):
            # 获取小批量数据集及其图像标签与域标签
            batch_mnist_image_data, batch_mnist_labels = train_source_datagen.__next__()  # train_source_datagen.next_batch()
            batch_mnist_m_image_data, batch_mnist_m_labels = train_target_datagen.__next__()  # train_target_datagen.next_batch()
            batch_domain_labels = np.vstack([np.tile([1., 0.], [len(batch_mnist_labels), 1]),
                                             np.tile([0., 1.], [len(batch_mnist_m_labels), 1])]).astype(np.float32)
            batch_image_data = np.concatenate([batch_mnist_image_data, batch_mnist_m_image_data], axis=0)
            # 更新学习率并可视化
            iter = (ep - 1) * train_iter_num + i
            process = iter * 1.0 / (self.cfg.epoch * train_iter_num)
            self.grl_lambd = grl_lambda_schedule(process)
            learning_rate = learning_rate_schedule(process, init_learning_rate=self.cfg.init_learning_rate)
            tf.keras.backend.set_value(self.optimizer.lr, learning_rate)
            with self.writer_hyperparameter.as_default():
                tf.summary.scalar("hyperparameter/learning_rate", tf.convert_to_tensor(learning_rate), iter)
                tf.summary.scalar("hyperparameter/grl_lambda", tf.convert_to_tensor(self.grl_lambd), iter)

            # 计算图像分类损失梯度
            with tf.GradientTape() as tape:
                # 计算图像分类预测输出、损失和精度
                image_cls_feature = self.feature_encoder(batch_mnist_image_data)
                image_cls_pred = self.image_cls_encoder(image_cls_feature,training=True)
                image_cls_loss = self.loss(batch_mnist_labels,image_cls_pred)
                image_cls_acc = self.acc(batch_mnist_labels, image_cls_pred)

                # 计算域分类预测输出、损失和精度
                domain_cls_feature = self.feature_encoder(batch_image_data)
                domain_cls_pred = self.domain_cls_encoder(self.grl(domain_cls_feature, self.grl_lambd),
                                                          training=True)
                domain_cls_loss = self.loss(batch_domain_labels, domain_cls_pred)
                domain_cls_acc = self.acc(batch_domain_labels, domain_cls_pred)

                # 计算训练损失、图像分类精度和域分类精度
                loss = tf.reduce_mean(image_cls_loss) + tf.reduce_mean(domain_cls_loss)
            # 自定义优化过程
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            # 计算平均损失与精度
            self.train_loss(loss)
            self.train_image_cls_loss(image_cls_loss)
            self.train_domain_cls_loss(domain_cls_loss)
            self.train_image_cls_acc(image_cls_acc)
            self.train_domain_cls_acc(domain_cls_acc)

            # 更新进度条
            self.progbar.update(i, [('loss', loss),
                               ('image_cls_loss', image_cls_loss),
                               ('domain_cls_loss', domain_cls_loss),
                               ("image_acc", image_cls_acc),
                               ("domain_acc", domain_cls_acc)])
        # 可视化损失与指标
        with self.writer_train.as_default():
            tf.summary.scalar("loss/loss", self.train_loss.result(), ep)
            tf.summary.scalar("loss/image_cls_loss", self.train_image_cls_loss.result(), ep)
            tf.summary.scalar("loss/domain_cls_loss", self.train_domain_cls_loss.result(), ep)
            tf.summary.scalar("acc/image_cls_acc", self.train_image_cls_acc.result(), ep)
            tf.summary.scalar("acc/domain_cls_acc", self.train_domain_cls_acc.result(), ep)

        return self.train_loss.result(),self.train_image_cls_acc.result()

    def eval_one_epoch(self,val_target_datagen,val_iter_num,ep):
        """
        这是一个周期的模型验证函数
        :param val_target_datagen: 目标域验证数据集生成器
        :param val_iter_num: 一个验证周期的迭代次数
        :param ep: 当前验证周期
        :return:
        """
        for i in np.arange(1, val_iter_num + 1):
            # 获取小批量数据集及其图像标签与域标签
            batch_mnist_m_image_data, batch_mnist_m_labels = val_target_datagen.__next__()
            batch_mnist_m_domain_labels = np.tile([0., 1.], [len(batch_mnist_m_labels), 1]).astype(np.float32)

            # 计算目标域数据的图像分类预测输出和域分类预测输出
            target_image_feature = self.feature_encoder(batch_mnist_m_image_data)
            target_image_cls_pred = self.image_cls_encoder(target_image_feature, training=False)
            target_domain_cls_pred = self.domain_cls_encoder(target_image_feature, training=False)

            # 计算目标域预测相关损失
            target_image_cls_loss = self.loss(batch_mnist_m_labels,target_image_cls_pred)
            target_domain_cls_loss = self.loss(batch_mnist_m_domain_labels,target_domain_cls_pred)
            target_loss = tf.reduce_mean(target_image_cls_loss) + tf.reduce_mean(target_domain_cls_loss)
            # 计算目标域图像分类精度
            image_cls_acc = self.acc(batch_mnist_m_labels, target_image_cls_pred)
            domain_cls_acc = self.acc(batch_mnist_m_domain_labels, target_domain_cls_pred)

            # 更新训练损失与训练精度
            self.val_loss(target_loss)
            self.val_image_cls_loss(target_image_cls_loss)
            self.val_domain_cls_loss(domain_cls_acc)
            self.val_image_cls_acc(image_cls_acc)
            self.val_domain_cls_acc(domain_cls_acc)

        # 可视化验证损失及其指标
        with self.writer_val.as_default():
            tf.summary.scalar("loss/loss", self.val_loss.result(), ep)
            tf.summary.scalar("loss/image_cls_loss", self.val_image_cls_loss.result(), ep)
            tf.summary.scalar("loss/domain_cls_loss", self.val_domain_cls_loss.result(), ep)
            tf.summary.scalar("acc/image_cls_acc", self.val_image_cls_acc.result(), ep)
            tf.summary.scalar("acc/domain_cls_acc", self.val_domain_cls_acc.result(), ep)
        return self.val_loss.result(), self.val_image_cls_acc.result()