# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 14:41
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : test_batch_images.py
# @Software: PyCharm

import os
import argparse
import numpy as np

from config.config import config
from model.MNIST2MNIST_M import MNIST2MNIST_M_DANN

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', help='This is the path of images.')
parser.add_argument('model_path', help="This is a ckpt file path of the DANN model ,it ends with '.ckpt'")
args = parser.parse_args()  # 类似于类的实例化，解析对象

# 初始化参数配置类和DANN模型
cfg = config()
dann = MNIST2MNIST_M_DANN(cfg)

# 初始化参数
image_dir = os.path.join(args.image_dir)
model_path = os.path.abspath(args.model_path)
# 获取批量图像路径
image_paths = [os.path.join(image_dir,image_name) for image_name in os.listdir(image_dir)]
# 测试批量图像
dann.test_image(image_paths,model_path)
