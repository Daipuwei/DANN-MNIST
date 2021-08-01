# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 下午12:33
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : model_utils.py
# @Software: PyCharm

import numpy as np

class EarlyStopping(object):
  """Stop training when a monitored metric has stopped improving.

  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.

  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.

  Arguments:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
  """

  def __init__(self,
               min_delta=0,
               patience=0,
               verbose=0,
               baseline=None):

    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.wait = 0
    self.stopped_epoch = 0
    self.monitor_op = np.less
    self.min_delta = min_delta
    self.stop_training = False
    self.best = np.Inf

  def on_train_begin(self):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf

  def on_epoch_end(self, epoch, val_loss):
    if self.monitor_op(val_loss - self.min_delta, self.best):
      self.best = val_loss
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.stop_training = True
        if self.verbose > 0:
            print('Epoch %05d: early stopping\n' % (self.stopped_epoch))
    return self.stop_training

def learning_rate_schedule(process,init_learning_rate = 0.01,alpha = 10.0 , beta = 0.75):
    """
    这个学习率的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param init_learning_rate: 初始学习率，默认为0.01
    :param alpha: 参数alpha，默认为10
    :param beta: 参数beta，默认为0.75
    """
    return init_learning_rate /(1.0 + alpha * process)**beta

def grl_lambda_schedule(process,gamma=10.0):
    """
    这是GRL的参数lambda的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param gamma: 参数gamma，默认为10
    """
    return 2.0 / (1.0+np.exp(-gamma*process)) - 1.0