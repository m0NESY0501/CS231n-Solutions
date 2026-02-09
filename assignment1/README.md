**Assignment1 — CS231n 实验展示**

- **作者**: m0NESY0501

**概述**

- 本目录包含 Assignment 1 的全部作业与实现。每题以独立 Notebook 形式保存，核心代码放在 `cs231n/` 包内，便于复现与批阅。

**📂 主要文件映射**

- **KNN (Q1)**: [assignment1/knn.ipynb](assignment1/knn.ipynb) — 最近邻分类器，核心实现： [cs231n/classifiers/k_nearest_neighbor.py](cs231n/classifiers/k_nearest_neighbor.py)
- **Softmax (Q2)**: [assignment1/softmax.ipynb](assignment1/softmax.ipynb) — 线性 softmax 分类器，核心实现： [cs231n/classifiers/softmax.py](cs231n/classifiers/softmax.py)
- **Two-layer Net (Q3)**: [assignment1/two_layer_net.ipynb](assignment1/two_layer_net.ipynb) — 两层神经网络实现（前向/反向/梯度检验），相关实现： [cs231n/classifiers/fc_net.py](cs231n/classifiers/fc_net.py) 或 notebook 内实现
- **Feature Design (Q4)**: [assignment1/features.ipynb](assignment1/features.ipynb) — 手工特征工程，修改点： [cs231n/features.py](cs231n/features.py)
- **Fully Connected Nets (Q5)**: [assignment1/FullyConnectedNets.ipynb](assignment1/FullyConnectedNets.ipynb) — 可扩展全连接网络，相关模块： [cs231n/layers.py](cs231n/layers.py)、[cs231n/layer_utils.py](cs231n/layer_utils.py)

**实现亮点**

- **数据处理**: 使用 `cs231n/data_utils.py` 加载 CIFAR-10，包含均值/方差标准化与训练/验证切分。
- **向量化与性能**: 关键运算（softmax、loss、backward）均使用 NumPy 向量化实现以提高速度。
- **数值稳定性**: softmax 实现采用减最大值技巧，交叉熵与正则化处理保持稳定。
- **验证工具**: 使用 `cs231n/gradient_check.py` 做数值梯度检查，确保反向传播实现正确。

**运行与依赖**

- 推荐 Python 包：

```powershell
pip install numpy scipy matplotlib jupyter
```

- 可选：将 notebook 导出为 HTML 便于离线展示：

```powershell
jupyter nbconvert --to html assignment1/softmax.ipynb
```
