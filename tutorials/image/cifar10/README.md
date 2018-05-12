**NOTE: For users interested in multi-GPU, we recommend looking at the newer [cifar10_estimator](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) example instead.**

---

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

---

**CIFAR-10 重点：**
- 核心数学组件包括卷积、修正线性激活、最大池化以及局部响应归一化；
- 训练过程中可视化网络行为，这些行为包括输入图像、损失情况、网络行为的分布情况以及梯度；
- 用于计算 学习参数的移动平均数 的教程，以及在评估期间使用这些平均数来提高预测性能。
- 实现了一种机制，使得学习率随着时间的推移而递减；
- 为输入数据设计预存取队列，将磁盘延迟和高开销的图像预处理操作与模型分离开来处理；

我们也提供了模型的多GUP版本，用以表明：
- 配置模型后，可以使其在多个GPU上并行训练
- 可以在多个GPU之间共享和更新变量值

---

Cifar-10 由60000张32*32的 RGB 彩色图片构成，共10个分类。
50000张训练，10000张测试（交叉验证）。
这个数据集最大的特点在于将识别迁移到了普适物体，而且应用于多分类（姊妹数据集Cifar-100达到100类，ILSVRC比赛则是1000类）

image的个数：60000
image的大小：32*32*3
class的个数：10 （飞机；汽车；鸟；猫；鹿；狗；青蛙；马；船；卡车）
注：卡车的类中只存在卡车，皮卡和suv都不在此类中。

train batch有5个，
test batch有1个，
batches.meta为class的名字，
每个batch的有10000个data和相对应的label，
label从0到9，每一个data的大小为3073个字节，等于32*32*3
