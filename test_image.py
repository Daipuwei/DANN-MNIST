# -*- coding: utf-8 -*-
# @Time    : 2020/2/17 14:40
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : test_image.py
# @Software: PyCharm

import os
import argparse

from config.config import config
from model.MNIST2MNIST_M import MNIST2MNIST_M_DANN
"""
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='This is a imgae path ')
parser.add_argument('model_path', help="This is a ckpt file path of the DANN model ,it ends with '.ckpt'")
args = parser.parse_args()  # 类似于类的实例化，解析对象
"""

def run_main():
    # 初始化参数配置类和DANN模型
    cfg = config()
    dann = MNIST2MNIST_M_DANN(cfg)

    # 初始化图像与权重路径
    image_path = "F:\\DANN-MNIST\\dataset\\Mnist2MnistM\\mnistM\\test\\7\\00000000_7.png"
    model_path = "C:\\Users\\77183\\Desktop\\DANN\\checkpoints\\20200220090516\\trained_model.ckpt"
    # 测试图像
    dann.test_image(os.path.abspath(image_path),os.path.abspath(model_path))

if __name__ == '__main__':
    run_main()
