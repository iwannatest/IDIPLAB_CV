# IDIPLAB_CV ![License](https://img.shields.io/aur/license/yaourt.svg)
IDIPLAB机器视觉工具包



## 概括

`IDIPLAB_CV`是

Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*

详细文档请阅读[wiki](https://github.com/Sandigal/IDIPLAB_CV/wiki)。



## 依赖环境
所有代码均基于`python3`编写，并且需要如下第三方库支持。对于`windows`系统，强烈建议通过`anaconda`进行环境配置。

- [python](https://www.python.org/)

* [tensorflow](https://www.tensorflow.org/): 详细信息请阅读[官方安装指南](https://www.tensorflow.org/install/)。
```bash
pip install --upgrade tensorflow
```

* [Keras](https://keras.io/): 详细信息请阅读[官方安装指南](https://keras.io/#installation)。
```bash
pip install keras -U --pre
```
* **可选依赖**:
  -  [CUDA](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu) 和 [cuDNN](http://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04) (通过GPU加速运算)
  - HDF5 和 [h5py](http://docs.h5py.org/en/latest/build.html) (将Keras保存到本地)
  - [graphviz](https://graphviz.gitlab.io/download/) 和 [pydot](https://github.com/erocarrera/pydot) ( 使用[visualization utilities](https://keras.io/visualization/)进行模型可视化)




## 未来计划
* 更多`augmentation`方法
* 支持多尺寸图像
