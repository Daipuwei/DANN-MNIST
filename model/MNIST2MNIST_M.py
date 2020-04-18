# -*- coding: utf-8 -*-
# @Time    : 2020/2/14 20:27
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : MNIST2MNIST_M.py
# @Software: PyCharm

import os
import cv2
import datetime
import numpy as np
import tensorflow as tf

from tensorflow import keras as K
from tensorflow.train import MomentumOptimizer

from utils.utils import plot_loss
from utils.utils import plot_accuracy
from utils.utils import AverageMeter
from utils.utils import make_summary

from model.GRL import GRL
from utils.utils import grl_lambda_schedule
from utils.utils import learning_rate_schedule

class MNIST2MNIST_M_DANN(object):

    def __init__(self,config):
        """
        这是MNINST与MNIST_M域适配网络的初始化函数
        :param config: 参数配置类
        """
        # 初始化参数类
        self.cfg = config

        # 定义相关占位符
        self.grl_lambd = tf.placeholder(tf.float32, [])                         # GRL层参数
        self.learning_rate = tf.placeholder(tf.float32, [])                     # 学习率
        self.source_image_labels = tf.placeholder(tf.float32, shape=(None, 10))
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 2))

        # 搭建深度域适配网络
        self.build_DANN()

        # 定义损失
        self.image_cls_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.source_image_labels,
                                                                          logits=self.image_cls))
        self.domain_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.domain_labels,
                                                                        logits=self.domain_cls))
        self.loss = self.image_cls_loss+self.domain_cls_loss

        # 定义精度
        correct_label_pred = tf.equal(tf.argmax(self.source_image_labels, 1), tf.argmax(self.image_cls, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

        # 定义模型保存类与加载类
        self.saver_save = tf.train.Saver(max_to_keep=100)  # 设置最大保存检测点个数为周期数

        # 定义学习率
        self.global_step = tf.Variable(tf.constant(0),trainable=False)
        #self.process = self.global_step / self.cfg.epoch

        # 初始化优化器
        #self.optimizer = MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        self.optimizer = MomentumOptimizer(self.learning_rate, momentum=self.cfg.momentum_rate)
        #var_list = [v.name() for v in tf.trainable_variables()]
        self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)


    def featur_extractor(self,image_input,name):
        """
        这是特征提取子网络的构建函数
        :param image_input: 图像输入张量
        :param name: 输出特征名称
        :return:
        """
        x = K.layers.Conv2D(filters=32,kernel_size=5,kernel_initializer=K.initializers.TruncatedNormal(stddev=0.1),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='relu')(image_input)
        x = K.layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
        x = K.layers.Conv2D(filters=48, kernel_size=5, kernel_initializer=K.initializers.TruncatedNormal(stddev=0.1),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='relu')(x)
        x = K.layers.MaxPool2D(pool_size=(2, 2),strides=2,name=name)(x)
        return x

    def build_image_classify_model(self,image_classify_feature):
        """
        这是搭建图像分类器模型的函数
        :param image_classify_feature: 图像分类特征张量
        :return:
        """
        # 搭建图像分类器
        x = K.layers.Lambda(lambda x:x,name="image_classify_feature")(image_classify_feature)
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(100,kernel_initializer=K.initializers.TruncatedNormal(stddev=0.1),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='relu')(x)
        #x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(10,kernel_initializer=K.initializers.TruncatedNormal(stddev=0.1),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='softmax',
                           name = "image_classify_pred")(x)
        return x

    def build_domain_classify_model(self,domain_classify_feature):
        """
        这是搭建域分类器的函数
        :param domain_classify_feature: 域分类特征张量
        :return:
        """
        # 搭建域分类器
        x = GRL(domain_classify_feature,self.grl_lambd)
        x = K.layers.Flatten()(x)
        x = K.layers.Dense(100,kernel_initializer=K.initializers.TruncatedNormal(stddev=0.01),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='relu')(x)
        #x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(2,kernel_initializer=K.initializers.TruncatedNormal(stddev=0.01),
                                bias_initializer = K.initializers.Constant(value=0.1), activation='softmax'
                           ,name="domain_classify_pred")(x)
        return x

    def build_DANN(self):
        """
        这是搭建域适配网络的函数
        :return:
        """
        # 定义源域、目标域的图像输入和DANN模型图像输入
        self.source_image_input = K.layers.Input(shape=self.cfg.image_input_shape,name="source_image_input")
        self.target_image_input = K.layers.Input(shape=self.cfg.image_input_shape,name="target_image_input")
        self.image_input = K.layers.Concatenate(axis=0,name="image_input")([self.source_image_input,self.target_image_input])
        self.image_input = (self.image_input - self.cfg.pixel_mean) / 255.0

        # 域分类器与图像分类器的共享特征
        share_feature = self.featur_extractor(self.image_input,"image_feature")

        # 均等划分共享特征为源域数据特征与目标域数据特征
        source_feature,target_feature = \
            K.layers.Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': 2})(share_feature)
        source_feature = K.layers.Lambda(lambda x:x,name="source_feature")(source_feature)

        # 获取图像分类结果和域分类结果张量
        self.image_cls = self.build_image_classify_model(source_feature)
        self.domain_cls = self.build_domain_classify_model(share_feature)

    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep):
        """
        这是评估模型在验证集上的性能的函数
        :param val_datagen: 验证集数据集生成器
        :param val_batch_num: 验证集数据集批量个数
        """
        epoch_loss_avg = AverageMeter()
        epoch_image_cls_loss_avg = AverageMeter()
        epoch_domain_cls_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()
        for i in np.arange(1, val_batch_num + 1):
            # 获取小批量数据集及其图像标签与域标签
            batch_mnist_m_image_data, batch_mnist_m_labels = val_datagen.__next__()#val_datagen.next_batch()
            batch_domain_labels = np.tile([0., 1.], [self.cfg.batch_size * 2, 1])

            #batch_mnist_m_image_data = (batch_mnist_m_image_data - self.cfg.val_image_mean) /255.0
            #batch_mnist_m_domain_labels = np.ones((self.cfg.batch_size,1))
            # 在验证阶段只利用目标域数据及其标签进行测试
            #batch_domain_labels = np.concatenate((batch_mnist_m_domain_labels, batch_mnist_m_domain_labels), axis=0)
            # 计算模型在验证集上相关指标的值
            val_loss, val_image_cls_loss, val_domain_cls_loss, val_acc = \
                sess.run([self.loss, self.image_cls_loss, self.domain_cls_loss, self.acc],
                        feed_dict={self.source_image_input: batch_mnist_m_image_data,
                                        self.target_image_input: batch_mnist_m_image_data,
                                        self.source_image_labels: batch_mnist_m_labels,
                                        self.domain_labels: batch_domain_labels})
            # 更新损失与精度的平均值
            epoch_loss_avg.update(val_loss, 1)
            epoch_image_cls_loss_avg.update(val_image_cls_loss, 1)
            epoch_domain_cls_loss_avg.update(val_domain_cls_loss, 1)
            epoch_accuracy.update(val_acc, 1)

        self.writer.add_summary(make_summary('val/val_loss', epoch_loss_avg.average),global_step=ep)
        self.writer.add_summary(make_summary('val/val_image_cls_loss', epoch_image_cls_loss_avg.average),global_step=ep)
        self.writer.add_summary(make_summary('val/val_domain_cls_loss', epoch_domain_cls_loss_avg.average),global_step=ep)
        self.writer.add_summary(make_summary('accuracy/val_accuracy', epoch_accuracy.average),global_step=ep)

        #self.writer1.add_summary(make_summary('val/val_loss', epoch_loss_avg.average),global_step=ep)
        #self.writer1.add_summary(make_summary('val/val_image_cls_loss', epoch_image_cls_loss_avg.average),global_step=ep)
        #self.writer1.add_summary(make_summary('val/val_domain_cls_loss', epoch_domain_cls_loss_avg.average),global_step=ep)
        #self.writer1.add_summary(make_summary('accuracy/val_accuracy', epoch_accuracy.average),global_step=ep)
        return epoch_loss_avg.average,epoch_image_cls_loss_avg.average,\
                   epoch_domain_cls_loss_avg.average,epoch_accuracy.average

    def train(self,train_source_datagen,train_target_datagen,val_datagen,pixel_mean,interval,
              train_iter_num,val_iter_num,pre_model_path=None):
        """
        这是DANN的训练函数
        :param train_source_datagen: 源域训练数据集生成器
        :param train_target_datagen: 目标域训练数据集生成器
        :param val_datagen: 验证数据集生成器
        :param interval: 验证间隔
        :param train_iter_num: 每个epoch的训练次数
        :param val_iter_num: 每次验证过程的验证次数
        :param pre_model_path: 预训练模型地址,与训练模型为ckpt文件，注意文件路径只需到.ckpt即可。
        """
        # 初始化相关文件目录路径
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(self.cfg.checkpoints_dir,time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        log_dir = os.path.join(self.cfg.logs_dir, time)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        result_dir = os.path.join(self.cfg.result_dir, time)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.cfg.save_config(time)

        # 初始化训练损失和精度数组
        train_loss_results = []                     # 保存训练loss值
        train_image_cls_loss_results = []           # 保存训练图像分类loss值
        train_domain_cls_loss_results = []          # 保存训练域分类loss值
        train_accuracy_results = []                 # 保存训练accuracy值

        # 初始化验证损失和精度数组，验证最大精度
        val_ep = []
        val_loss_results = []                     # 保存验证loss值
        val_image_cls_loss_results = []           # 保存验证图像分类loss值
        val_domain_cls_loss_results = []          # 保存验证域分类loss值
        val_accuracy_results = []                 # 保存验证accuracy值
        val_acc_max = 0                           # 最大验证精度

        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            if pre_model_path is not None:              # pre_model_path的地址写到.ckpt
                saver_restore = tf.train.import_meta_graph(pre_model_path+".meta")
                saver_restore.restore(sess,pre_model_path)
                print("restore model from : %s" % (pre_model_path))

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, sess.graph)
            #self.writer1 = tf.summary.FileWriter(os.path.join("./tf_dir"), sess.graph)

            print('\n----------- start to train -----------\n')

            total_global_step = self.cfg.epoch * train_iter_num
            for ep in np.arange(self.cfg.epoch):
                # 初始化每次迭代的训练损失与精度平均指标类
                epoch_loss_avg = AverageMeter()
                epoch_image_cls_loss_avg = AverageMeter()
                epoch_domain_cls_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

                # 初始化精度条
                progbar = K.utils.Progbar(train_iter_num)
                print('Epoch {}/{}'.format(ep+1, self.cfg.epoch))
                batch_domain_labels = np.vstack([np.tile([1., 0.], [self.cfg.batch_size // 2, 1]),
                                           np.tile([0., 1.], [self.cfg.batch_size // 2, 1])])
                for i in np.arange(1,train_iter_num+1):
                    # 获取小批量数据集及其图像标签与域标签
                    batch_mnist_image_data, batch_mnist_labels = train_source_datagen.__next__()#train_source_datagen.next_batch()
                    batch_mnist_m_image_data, batch_mnist_m_labels = train_target_datagen.__next__()#train_target_datagen.next_batch()
                    """
                    print(np.shape(batch_mnist_image_data))
                    print(np.shape(batch_mnist_labels))
                    print(np.shape(batch_mnist_domain_labels))
                    print(np.shape(batch_mnist_m_image_data))
                    print(np.shape(batch_mnist_m_labels))
                    print(np.shape(batch_mnist_m_domain_labels))
                    """
                    # 计算学习率和GRL层的参数lambda
                    global_step = (ep-1)*train_iter_num + i
                    process = global_step * 1.0 / total_global_step
                    leanring_rate = learning_rate_schedule(process,self.cfg.init_learning_rate)
                    grl_lambda = grl_lambda_schedule(process)

                    # 前向传播,计算损失及其梯度
                    op,train_loss,train_image_cls_loss,train_domain_cls_loss,train_acc = \
                        sess.run([self.train_op,self.loss,self.image_cls_loss,self.domain_cls_loss,self.acc],
                                  feed_dict={self.source_image_input:batch_mnist_image_data,
                                             self.target_image_input:batch_mnist_m_image_data,
                                             self.source_image_labels:batch_mnist_labels,
                                             self.domain_labels:batch_domain_labels,
                                             self.learning_rate:leanring_rate,
                                             self.grl_lambd:grl_lambda})
                    self.writer.add_summary(make_summary('learning_rate', leanring_rate),global_step=global_step)
                    #self.writer1.add_summary(make_summary('learning_rate', leanring_rate), global_step=global_step)

                    # 更新训练损失与训练精度
                    epoch_loss_avg.update(train_loss,1)
                    epoch_image_cls_loss_avg.update(train_image_cls_loss,1)
                    epoch_domain_cls_loss_avg.update(train_domain_cls_loss,1)
                    epoch_accuracy.update(train_acc,1)

                    # 更新进度条
                    progbar.update(i, [('train_image_cls_loss', train_image_cls_loss),
                                       ('train_domain_cls_loss', train_domain_cls_loss),
                                       ('train_loss', train_loss),
                                       ("train_acc",train_acc)])

                # 保存相关损失与精度值，可用于可视化
                train_loss_results.append(epoch_loss_avg.average)
                train_image_cls_loss_results.append(epoch_image_cls_loss_avg.average)
                train_domain_cls_loss_results.append(epoch_domain_cls_loss_avg.average)
                train_accuracy_results.append(epoch_accuracy.average)

                self.writer.add_summary(make_summary('train/train_loss', epoch_loss_avg.average),global_step=ep+1)
                self.writer.add_summary(make_summary('train/train_image_cls_loss', epoch_image_cls_loss_avg.average),
                                   global_step=ep+1)
                self.writer.add_summary(make_summary('train/train_domain_cls_loss', epoch_domain_cls_loss_avg.average),
                                   global_step=ep+1)
                self.writer.add_summary(make_summary('accuracy/train_accuracy', epoch_accuracy.average),global_step=ep+1)

                #self.writer1.add_summary(make_summary('train/train_loss', epoch_loss_avg.average),global_step=ep+1)
                #self.writer1.add_summary(make_summary('train/train_image_cls_loss', epoch_image_cls_loss_avg.average),
                #                   global_step=ep+1)
                #self.writer1.add_summary(make_summary('train/train_domain_cls_loss', epoch_domain_cls_loss_avg.average),
                #                   global_step=ep+1)
                #self.writer1.add_summary(make_summary('accuracy/train_accuracy', epoch_accuracy.average),global_step=ep+1)

                if (ep+1) % interval == 0:
                    # 评估模型在验证集上的性能
                    val_ep.append(ep)
                    val_loss, val_image_cls_loss,val_domain_cls_loss, \
                        val_accuracy = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1)
                    val_loss_results.append(val_loss)
                    val_image_cls_loss_results.append(val_image_cls_loss)
                    val_domain_cls_loss_results.append(val_domain_cls_loss)
                    val_accuracy_results.append(val_accuracy)
                    str =  "Epoch{:03d}_val_image_cls_loss{:.3f}_val_domain_cls_loss{:.3f}_val_loss{:.3f}" \
                           "_val_accuracy{:.3%}".format(ep+1,val_image_cls_loss,val_domain_cls_loss,val_loss,val_accuracy)
                    print(str)

                    if val_accuracy > val_acc_max:              # 验证精度达到当前最大，保存模型
                        val_acc_max = val_accuracy
                        self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))

            # 保存训练与验证结果
            path = os.path.join(result_dir, "train_loss.jpg")
            plot_loss(np.arange(1,len(train_loss_results)+1), [np.array(train_loss_results),
                                np.array(train_image_cls_loss_results),np.array(train_domain_cls_loss_results)],
                                path, "train")
            path = os.path.join(result_dir, "val_loss.jpg")
            plot_loss(np.array(val_ep)+1, [np.array(val_loss_results),
                                np.array(val_image_cls_loss_results),np.array(val_domain_cls_loss_results)],
                               path, "val")
            train_acc = np.array(train_accuracy_results)[np.array(val_ep)]
            path = os.path.join(result_dir, "accuracy.jpg")
            plot_accuracy(np.array(val_ep)+1, [train_acc, val_accuracy_results], path)

            # 保存最终的模型
            model_path = os.path.join(checkpoint_dir,"trained_model.ckpt")
            self.saver_save.save(sess,model_path)
            print("Train model finshed. The model is saved in : ", model_path)
            print('\n----------- end to train -----------\n')

    def test_image(self,image_path,model_path):
        """
        这是测试一张图像的函数
        :param image_path: 图像路径
        :param model_path: 模型路径
        :return:
        """
        # 读取图像数据，并进行数组维度扩充
        image = cv2.imread(image_path)
        image = np.expand_dims(image,axis=0)

        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            saver_restore = tf.train.import_meta_graph(model_path+".meta")
            saver_restore.restore(sess, model_path)

            # 进行测试
            img_cls_pred = sess.run([self.image_cls],feed_dict={self.source_image_input: image})
            pred_label = np.argmax(img_cls_pred[0])+1
            print("%s is %d" %(image_path,pred_label))

    def test_batch_images(self, image_paths, model_path):
        """
        这是测试一张图像的函数
        :param image_paths: 图像路径数组
        :param model_path: 模型路径
        :return:
        """
        # 批量读取图像数据
        images = np.array([cv2.imread(image_path) for image_path in image_paths])

        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            saver_restore = tf.train.import_meta_graph(model_path+".meta")
            saver_restore.restore(sess, model_path)

            # 进行测试
            img_cls_pred = sess.run([self.image_cls], feed_dict={self.source_image_input: images})
            pred_label = np.argmax(img_cls_pred,axis=0) + 1
            for i,image_path in enumerate(image_paths):
                print("%s is %d" % (image_path, pred_label[i]))
