# CS231n — Assignment1 & Assignment2 汇总

## 项目概述

- 该仓库包含我为 Stanford CS231n 课程完成的 Assignment 1（全连接神经网络与数值梯度检验等基础实现）和 Assignment 2（卷积网络、BatchNorm、Dropout 与部分 PyTorch 练习）的代码与笔记本。本文档为面向导师的简要说明，概述工作目标、实现要点、如何运行以及重要文件位置。

## 包含内容（高层）

- 课程笔记本：`ConvolutionalNetworks.ipynb`、`BatchNormalization.ipynb`、`Dropout.ipynb`、`PyTorch.ipynb`、`RNN_Captioning_pytorch.ipynb`。
- 核心实现（纯 NumPy + 少量 Cython 优化）：`cs231n/` 内的模型、层实现、优化器与工具脚本。

## Assignment 1（关键点）

- 目标：实现分类器训练的基础模块，包括线性层、激活、softmax/SVM loss、反向传播与数值梯度检验。
- 主要实现文件：[cs231n/classifiers/fc_net.py](cs231n/classifiers/fc_net.py)、[cs231n/layers.py](cs231n/layers.py)、[cs231n/gradient_check.py](cs231n/gradient_check.py)、[cs231n/solver.py](cs231n/solver.py)。
- 我在 `fc_net.py` 中实现了两层/多层全连接网络的初始化、前向与反向，使用 `gradient_check.py` 做了数值梯度验证以确保求导实现正确。

## Assignment 2（关键点）

- 目标：将前面的网络扩展到卷积神经网络（CNN），实现卷积/池化层、Batch Normalization、Dropout，并使用这些组件训练简单的 CNN 模型完成图像分类任务。
- 主要实现文件：[cs231n/classifiers/cnn.py](cs231n/classifiers/cnn.py)、[cs231n/layers.py](cs231n/layers.py)、[cs231n/im2col.py](cs231n/im2col.py)、`im2col_cython`（加速实现）。
- 笔记本中有独立实验：`Dropout.ipynb`（dropout 实验与可视化）、`BatchNormalization.ipynb`（BN 实验）、`ConvolutionalNetworks.ipynb`（CNN 架构设计与训练流程）。

## 如何运行（快速指南）

1. 安装依赖（推荐虚拟环境）：

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt  # see Dependencies 下的说明
```

1. 设置 `PYTHONPATH`（在仓库根目录）：

```bash
set PYTHONPATH=%CD%
```

1. 打开并运行笔记本：

```bash
jupyter notebook ConvolutionalNetworks.ipynb
jupyter notebook BatchNormalization.ipynb
```

1. 训练脚本 / 单元测试：仓库中的 `solver.py` 与笔记本内训练单元可直接调用实现的网络与优化器。

## 依赖（建议）

- Python 3.8+，NumPy，SciPy，matplotlib，jupyter。若要编译 Cython 加速模块需要 `cython` 与合适的编译工具链。处理 PyTorch 练习还需安装 `torch`。

## 关键文件速览

- 笔记本：`ConvolutionalNetworks.ipynb`, `BatchNormalization.ipynb`, `Dropout.ipynb`, `PyTorch.ipynb`, `RNN_Captioning_pytorch.ipynb`。
- 核心代码：`cs231n/classifiers/fc_net.py`, `cs231n/classifiers/cnn.py`, `cs231n/layers.py`, `cs231n/solver.py`, `cs231n/optim.py`。
- 辅助工具：`cs231n/gradient_check.py`, `cs231n/im2col.py`, `im2col_cython.*`。

## 实验与结果（摘要）

- 在笔记本中我完成了下列对比实验：不同 dropout 比例对训练/验证精度影响；应用 BatchNorm 后的收敛速度改善；多个小型 CNN 架构在 CIFAR-10 子集上的训练曲线（可在相应笔记本中查看图表和数值）。具体图表与训练日志保存在对应的 notebook 输出单元中。

## 博客与记录

- 个人学习记录与实验日志发布在知乎专栏：<https://www.zhihu.com/column/c_1988741880209507029>
