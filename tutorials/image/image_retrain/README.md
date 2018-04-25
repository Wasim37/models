此示例显示如何使用TensorFlow Hub对图像分类器进行再训练。

图像识别模型通常具有数百万个参数，如果从头开始对它们进行训练需要大量的标记数据和计算能力。
迁移学习是一种技巧，它使用一个已经在相关任务上接受过训练的模型，并在新模型中重新使用该模型来简化大部分工作。
在本教程中，我们将重用在ImageNet上训练好的功能强大的图像分类器中的特征提取功能，并简单地在上面训练一个新的分类图层。

- 教程简介：https://www.tensorflow.org/tutorials/image_retraining
- TensorFlow Hub 官网：https://www.tensorflow.org/hub/
- TensorFlow Hub 简介：http://geek.csdn.net/news/detail/257888