# DANN-MNIST-tf2
这是论文[Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495)的复现代码，完成了MNIST与MNIST-M数据集之间的迁移训练

# 实验环境

 1. tensorflow=2.4.0
 2. opencv
 3. numpy
 4. pickle
 5. skimage

# 文档结构
- `checkpoints`存放训练过程中模型权重；
- `logs`存放模型训练过程中相关日志文件；
- `config`存放参数配置类脚本及训练过程中参数配置文件；
- `model`存放网络模型定义脚本；
- `model_data`存放包括但不限于数据集、预训练模型等文件；
- `utils`存放包括但不限于数据集和模型训练相关工具类和工具脚本；
- `image`存放tensorboard可视化截图；
- `create_mnistm.py`是根据MNIST数据集生成MNIST-M数据集的脚本；
- `train_MNIST2MNIST_M.py`是利用MNIST和MNIST-M数据集进行DANN自适应模型训练的脚本；


# How to train

 首先下载BSDS500数据集 ，放在`model_data/dataset`路径下。其下载路径如下：
 - 官网：[BSDS500数据集](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) 
 -  github：[BSDS500数据集](https://github.com/Daipuwei/DANN-MNIST-tf2/releases/download/master/BSR_bsds500.zip) 

 
然后执行`python create_mnistm.py`生成MNIST-M数据集，根据自己需要修改`create_mnistm.py`中`BST_PATH`、`mnist_dir`和`mnistm _dir
`，默认路径如下：
```python
BST_PATH = os.path.abspath('./model_data/dataset/BSR_bsds500.tgz')
mnist_dir = os.path.abspath("model_data/dataset/MNIST")
mnistm_dir = os.path.abspath("model_data/dataset/MNIST_M")
```
最后运行如下命令进行MNIST和MNIST-M数据集之间的自适应模型训练，根据自己的需要进行修改相关超参数，例如`init_learning_rate`、`momentum_rate`、`batch_size`、`epoch`、`pre_model_path`、`source_dataset_path`和`target_dataset_path`。
```python
python train_MNIST2MNIST_M.py
```



#  实验结果
下面主要包括了MNIST和MNIST-M数据集在自适应训练过程中**学习率**、**梯度反转层参数**$\lambda$、训练集和验证集的**图像分类损失**、**域分类损失**、**图像分类精度**、**域分类精度**和**模型总损失**的可视化。

首先是超参数**学习率**和**梯度反转层参数**$\lambda$在训练过程中的数据可视化。

![超参数可视化](https://github.com/Daipuwei/DANN-MNIST-tf2/blob/master/image/hyperparameter.png#pic_center)

接着是训练数据集和验证数据集的**图像分类精度**和**域分类精度**在训练过程中的数据可视化，其中蓝色代表训练集，红色代表验证集。

![指标可视化](https://github.com/Daipuwei/DANN-MNIST-tf2/blob/master/image/acc.png#pic_center)

最后是训练数据集和验证数据集的**图像分类损失**和**域分类损失**在训练过程中的数据可视化，其中蓝色代表训练集，红色代表验证集。

![损失可视化](https://github.com/Daipuwei/DANN-MNIST-tf2/blob/master/image/loss.png#pic_center)


#  相关博客资料

 CSDN博客链接：

 1. [【深度域适配】一、DANN与梯度反转层（GRL）详解](https://daipuweiai.blog.csdn.net/article/details/104478550)
 2. 【[深度域适配】二、利用DANN实现MNIST和MNIST-M数据集迁移训练](https://daipuweiai.blog.csdn.net/article/details/104495520)

知乎专栏链接：
 1. [【深度域适配】一、DANN与梯度反转层（GRL）详解](https://zhuanlan.zhihu.com/p/109051269)
 2. 【[深度域适配】二、利用DANN实现MNIST和MNIST-M数据集迁移训练](https://zhuanlan.zhihu.com/p/109057360)
