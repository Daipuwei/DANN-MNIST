# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 15:05
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : config.py
# @Software: PyCharm

import os

class config(object):

    __defualt_dict__ = {
        "pre_model_path":None,
        "checkpoints_dir":os.path.abspath("./checkpoints"),
        "logs_dir":os.path.abspath("./logs"),
        "config_dir":os.path.abspath("./config"),
        "dataset_dir": os.path.abspath("./dataset"),
        #"dataset_dir": os.path.abspath("/input0"),
        "result_dir": os.path.abspath("./result"),
        "image_input_shape":(28,28,3),
        "image_size":28,
        "init_learning_rate": 1e-2,
        "momentum_rate": 0.9,
        "batch_size":64,
        "epoch":500,
    }

    def __init__(self,**kwargs):
        """
        这是参数配置类的初始化函数
        :param kwargs: 参数字典
        """
        # 初始化相关配置参数
        self.__dict__.update(self. __defualt_dict__)
        # 根据相关传入参数进行参数更新
        self.__dict__.update(kwargs)

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def set(self,**kwargs):
        """
        这是参数配置的设置函数
        :param kwargs: 参数字典
        :return:
        """
        # 根据相关传入参数进行参数更新
        self.__dict__.update(kwargs)

    def save_config(self,time):
        """
        这是保存参数配置类的函数
        :param time: 时间点字符串
        :return:
        """
        # 更新相关目录
        self.checkpoints_dir = os.path.join(self.checkpoints_dir,time)
        self.logs_dir = os.path.join(self.logs_dir,time)
        self.config_dir = os.path.join(self.config_dir,time)
        self.result_dir = os.path.join(self.result_dir,time)

        if not os.path.exists(self.config_dir):
            os.mkdir(self.config_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        config_txt_path = os.path.join(self.config_dir,"config.txt")
        with open(config_txt_path,'a') as f:
            for key,value in self.__dict__.items():
                if key in ["checkpoints_dir","logs_dir","config_dir"]:
                    value = os.path.join(value,time)
                    s = key+": "+value+"\n"
                    f.write(s)
