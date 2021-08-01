# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 下午12:55
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : GradientReveresalLayer.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.custom_gradient
def gradient_reversal(x,alpha=1.0):
	def grad(dy):
		return -dy * alpha, None
	return x, grad

class GradientReversalLayer(Layer):

	def __init__(self,**kwargs):
		"""
		这是梯度反转层的初始化函数
		:param kwargs: 参数字典
		"""
		super(GradientReversalLayer,self).__init__(kwargs)

	def call(self, x,alpha=1.0):
		"""
		这是梯度反转层的初始化函数
		:param x: 输入张量
		:param alpha: alpha系数，默认为1
		:return:
		"""
		return gradient_reversal(x,alpha)