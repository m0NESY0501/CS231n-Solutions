# CS231n — Assignment1 & Assignment2 汇总

> **✅ 完成状态**：Assignment 1 与 Assignment 2 均已全部完成。

## 项目概述

- 该仓库包含我为 Stanford CS231n 课程完成的 **Assignment 1**（KNN、Softmax、全连接神经网络与数值梯度检验等基础实现）和 **Assignment 2**（卷积网络、BatchNorm、Dropout、PyTorch 练习与 RNN 图像描述）的代码与笔记本。本文档为面向导师的简要说明，概述工作目标、实现要点、如何运行以及重要文件位置。

## 包含内容（高层）

- **Assignment 1 笔记本**：`assignment1/knn.ipynb`、`assignment1/softmax.ipynb`、`assignment1/two_layer_net.ipynb`、`assignment1/features.ipynb`、`assignment1/FullyConnectedNets.ipynb`。
- **Assignment 2 笔记本**：`assignment2/ConvolutionalNetworks.ipynb`、`assignment2/BatchNormalization.ipynb`、`assignment2/Dropout.ipynb`、`assignment2/PyTorch.ipynb`、`assignment2/RNN_Captioning_pytorch.ipynb`。
- **核心实现**（纯 NumPy + 少量 Cython 优化 + PyTorch）：各 `assignment*/cs231n/` 内的模型、层实现、优化器与工具脚本。

## Assignment 1（关键点）

- **目标**：实现分类器训练的基础模块，包括 KNN、Softmax、线性层、激活、softmax/SVM loss、反向传播与数值梯度检验。
- **Q1 KNN**：[assignment1/cs231n/classifiers/k_nearest_neighbor.py](assignment1/cs231n/classifiers/k_nearest_neighbor.py)
- **Q2 Softmax**：[assignment1/cs231n/classifiers/softmax.py](assignment1/cs231n/classifiers/softmax.py)
- **Q3 Two-layer Net**：两层神经网络（前向/反向/梯度检验）
- **Q4 Features**：[assignment1/cs231n/features.py](assignment1/cs231n/features.py)
- **Q5 Fully Connected Nets**：[assignment1/cs231n/classifiers/fc_net.py](assignment1/cs231n/classifiers/fc_net.py)、[assignment1/cs231n/layers.py](assignment1/cs231n/layers.py)、[assignment1/cs231n/gradient_check.py](assignment1/cs231n/gradient_check.py)、[assignment1/cs231n/solver.py](assignment1/cs231n/solver.py)

## Assignment 2（关键点）

- **目标**：将前面的网络扩展到卷积神经网络（CNN），实现卷积/池化层、Batch Normalization、Dropout，使用 PyTorch 进行 CNN 训练，并实现 RNN 图像描述（Image Captioning）。
- **CNN 实现**：[assignment2/cs231n/classifiers/cnn.py](assignment2/cs231n/classifiers/cnn.py)、[assignment2/cs231n/layers.py](assignment2/cs231n/layers.py)、[assignment2/cs231n/im2col.py](assignment2/cs231n/im2col.py)、`im2col_cython`（加速实现）。
- **PyTorch & RNN**：[assignment2/cs231n/classifiers/rnn_pytorch.py](assignment2/cs231n/classifiers/rnn_pytorch.py)、[assignment2/cs231n/rnn_layers_pytorch.py](assignment2/cs231n/rnn_layers_pytorch.py)、[assignment2/cs231n/captioning_solver_pytorch.py](assignment2/cs231n/captioning_solver_pytorch.py)
- **笔记本实验**：`Dropout.ipynb`（dropout 实验与可视化）、`BatchNormalization.ipynb`（BN 实验）、`ConvolutionalNetworks.ipynb`（CNN 架构设计与训练流程）、`RNN_Captioning_pytorch.ipynb`（RNN 图像描述）。

## 如何运行（快速指南）

1. 安装依赖（推荐虚拟环境）：

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt  # see Dependencies 下的说明
```

2. 设置 `PYTHONPATH`（运行哪个 assignment 就指向其目录，使 `cs231n` 模块可导入）：

```bash
# Windows CMD（以 assignment2 为例）
cd assignment2
set PYTHONPATH=%CD%
```

3. 打开并运行笔记本（在仓库根目录下）：

```bash
# Assignment 1
jupyter notebook assignment1/knn.ipynb
jupyter notebook assignment1/FullyConnectedNets.ipynb

# Assignment 2
jupyter notebook assignment2/ConvolutionalNetworks.ipynb
jupyter notebook assignment2/BatchNormalization.ipynb
jupyter notebook assignment2/RNN_Captioning_pytorch.ipynb
```

4. 训练脚本 / 单元测试：各 assignment 中的 `solver.py` 与笔记本内训练单元可直接调用实现的网络与优化器。

## 依赖（建议）

- Python 3.8+，NumPy，SciPy，matplotlib，jupyter。若要编译 Cython 加速模块需要 `cython` 与合适的编译工具链。处理 PyTorch 练习还需安装 `torch`。

## 关键文件速览

- **Assignment 1 笔记本**：`assignment1/knn.ipynb`, `assignment1/softmax.ipynb`, `assignment1/two_layer_net.ipynb`, `assignment1/features.ipynb`, `assignment1/FullyConnectedNets.ipynb`
- **Assignment 2 笔记本**：`assignment2/ConvolutionalNetworks.ipynb`, `assignment2/BatchNormalization.ipynb`, `assignment2/Dropout.ipynb`, `assignment2/PyTorch.ipynb`, `assignment2/RNN_Captioning_pytorch.ipynb`
- **核心代码**：`assignment1/cs231n/classifiers/fc_net.py`, `assignment2/cs231n/classifiers/cnn.py`, `assignment2/cs231n/classifiers/rnn_pytorch.py`, `assignment1/cs231n/layers.py`, `assignment2/cs231n/layers.py`, `assignment2/cs231n/solver.py`, `assignment2/cs231n/optim.py`
- **辅助工具**：`assignment1/cs231n/gradient_check.py`, `assignment2/cs231n/gradient_check.py`, `assignment2/cs231n/im2col.py`, `im2col_cython.*`

## 实验与结果（摘要）

- **Assignment 1**：KNN、Softmax、Two-layer Net、特征工程、全连接网络在 CIFAR-10 上的分类实验与梯度检验。
- **Assignment 2**：不同 dropout 比例对训练/验证精度影响；应用 BatchNorm 后的收敛速度改善；多个小型 CNN 架构在 CIFAR-10 子集上的训练曲线；PyTorch CNN 训练；RNN 图像描述（COCO 数据集）。具体图表与训练日志保存在对应的 notebook 输出单元中。

## 博客与记录

- 个人学习记录与实验日志发布在知乎专栏：<https://www.zhihu.com/column/c_1988741880209507029>
