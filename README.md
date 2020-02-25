# DANN-MNIST
这是论文[Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495)的复现代码，并完成了MNIST与MNIST-M数据集之间的迁移训练

# 实验环境

 1. tensorflow 1.14.0
 2. opencv 3.4.5.20
 3. numpy 1.18.1

---
# Train
首先下载[MNIST数据集](http://yann.lecun.com/exdb/mnist/)，放在项目文件的/data/mnist子文件夹下。之后下载[BSDS500数据集](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500)放在项目文件的/data/BSR_bsds500.tgz路径下。

之后为了生成MNSIT-M数据集，运行create_mnistm.py脚本，命令如下：

```python
python create_mnistm.py
```
脚本运行结束后，MNIST-M数据集将保存在项目文件的/data/mnistm子文件夹下。
之后运行模型训练脚本train.py即可，运行命令为为：
```python
python train.py
```
---
# 实验结果
下面是训练过程中的相关tensorboard的相关指标在训练过程中的走势图。首先是训练误差的走势图，主要包括训练域分类误差、训练图像分类误差和训练总误差。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200225145253405.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)
接下来是验证误差的走势图，主要包括验证域分类误差、验证图像分类误差和验证练总误差。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200225145155223.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)
然后是训练过程中学习率的走势图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200225145450917.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)
最后是精度走势图，主要包括训练精度和测试精度。**其中训练精度是在源域数据集即MNIST数据集上的统计结果，验证精度是在目标域数据集即MNIST-M数据集上的统计结果**。从图中可以看出，DANN在训练MNIST-M数据集时没有使用对应的标签，MNSIT-M数据集上的精度最终收敛到75.4%，效果相比于81.49%还有一定距离，但鉴于没有使用任何数据增强和dropout，这个结果可以接受。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022514570874.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)
