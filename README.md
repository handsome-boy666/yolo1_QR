# YOLOv1_QR Project

本项目基于 YOLOv1 改进用于二维码检测，涵盖训练、评估、预测及实时摄像头检测功能。原版 YOLOv1 支持 20 类目标检测，但在某些特定场景中我们只需检测定位一种物体，因此本项目对yolo1网络进行了修改和简化，对于初学者是一个很令人惊喜的实践。

## 1. YOLOv1 原理简介

YOLOv1 是一种单阶段目标检测算法，其核心思想是将目标检测问题转化为回归问题。
### 1.1 YOLOv1 网络结构

![alt text](images/image1.png)

*   **网格划分 (Grid Split)**: 将输入图像划分为 $7 \times 7$ 的网格。如果一个物体的中心落在某个网格内，该网格就负责检测该物体。
*   **边界框回归 (Bounding Box Regression)**: 每个网格预测 $2$ 个边界框 (Bounding Box)。每个边界框包含 5 个预测值： $x, y, w, h$  (位置与大小) 和 $confidence$ (置信度)。
    *   $x, y$: 边界框中心相对于网格单元的偏移。
    *   $w, h$: 边界框相对于整张图像的宽高。
    *   $confidence$: 预测框包含物体的概率 $\times$ 预测框与真实框的 IoU。
*   **类别概率 (Class Probability)**: 每个网格还需要预测 $20$ 个类别的条件概率。
    * 分析：每个网格需要得到 $2$ 个边界框的预测值，每个边界框包含 $5$ 个值，加上 $20$ 个每一类的概率，总共是 $2 \times 5 + 20 = 30$ 个值。可以看到符合最终输出的 $7 \times 7 \times 30$ 的格式。
<p align="center">
  <img src="images/image2.png" alt="image2" width="300"/>
</p>

### 1.2 YOLOv1 损失函数
$$
Loss = \lambda_{coord} \cdot Loss_{coord} + \lambda_{noobj} \cdot Loss_{noobj} + Loss_{conf} + Loss_{class}
$$

*  **四大损失项分别对应：**
    - 边界框坐标损失 $Loss_{coord}$ ，权重一般为 $\lambda_{coord} = 5$
    - 无目标置信度损失 $Loss_{noobj}$ ，权重一般为 $\lambda_{noobj} = 0.5$
    - 有目标置信度损失 $Loss_{conf}$ ，权重一般为 $1$
    - 类别损失 $Loss_{class}$ ，权重一般为 $1$
1. **边界框坐标损失：**

$$
Loss_{coord} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{obj} \left[
(x_{ij} - \hat{x}_{ij})^2 + (y_{ij} - \hat{y}_{ij})^2 +
(\sqrt{w_{ij}} - \sqrt{\hat{w}_{ij}})^2 + (\sqrt{h_{ij}} - \sqrt{\hat{h}_{ij}})^2
\right]
$$

2. **无目标置信度损失：**

$$
Loss_{noobj} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{noobj} (C_{ij} - \hat{C}_{ij})^2
$$

3. **有目标置信度损失：**

$$
Loss_{conf} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{I}_{ij}^{obj} (C_{ij} - \hat{C}_{ij})^2
$$

4. **类别损失：**

$$
Loss_{class} = \sum_{i=0}^{S^2} \sum_{c \in classes} \mathbb{I}_{i}^{obj} (p_{i,c} - \hat{p}_{i,c})^2
$$

得到最终损失函数

$$
\begin{aligned}
Loss &= \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\
&+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] \\
&+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 \\
&+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 \\
&+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
\end{aligned}
$$

其中：
*   $\mathbb{1}_{ij}^{obj}$ 表示第 $i$ 个网格的第 $j$ 个 bbox 负责预测物体。
*   $\lambda_{coord} = 5$, $\lambda_{noobj} = 0.5$。

**本项目改进**:
针对二维码检测任务（单类别），简化了部分结构，专注于检测二维码的位置。

## 2. 本项目简介

本项目使用 PyTorch 实现，针对二维码数据集进行训练。
*   **模型**: 简化的 YOLOv1 结构，专用于单类别（QR Code）检测。
*   **数据**: 数据集应包含图片和对应的标签（YOLO 格式）。
*   **功能**: 支持断点续训、测试集指标评估 (mIoU, Precision, Recall)、可视化预测结果、摄像头实时检测与录制。

### 本项目 (YOLOv1_QR) 损失函数公式

由于本项目仅检测二维码（单类别，C=0 或隐含为 1 但不计算分类损失），且每个网格仅预测 1 个边界框 ($B=1$)，公式简化如下：

$$
\begin{aligned}
Loss &= \lambda_{coord} \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\
&+ \lambda_{coord} \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] \\
&+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} (C_i - \hat{C}_i)^2 \\
&+ \lambda_{noobj} \sum_{i=0}^{S^2} \mathbb{1}_{i}^{noobj} (C_i - \hat{C}_i)^2
\end{aligned}
$$

**主要区别**:
1.  **移除分类损失**: 删除了最后一项 $\sum (p_i(c) - \hat{p}_i(c))^2$，因为只有一类（QR Code）。
2.  **单框预测**: 每个网格只预测一个框 ($B=1$)，简化了 $j$ 的求和。
3.  **数值稳定性**: 代码中对 $w, h$ 开根号前进行了 `clamp(min=1e-6)` 处理，防止 NaN。

## 3. 环境配置

建议使用 Anaconda 创建虚拟环境。

**依赖库**:
*   Python >= 3.8
*   PyTorch (建议 GPU 版本)
*   torchvision
*   opencv-python
*   Pillow
*   tqdm
*   PyYAML

**安装命令示例**:
```bash
conda create -n yolo_qr python=3.9
conda activate yolo_qr
# 安装 PyTorch (请根据官网选择适合你 CUDA 版本的命令)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# 安装其他依赖
pip install opencv-python pillow tqdm pyyaml
```

## 4. 快速开始

### 4.1 配置文件 (config.yaml)

项目的所有主要配置都在 `config.yaml` 中。运行程序前，请确保配置文件中的路径（如 `data_dir`, `ckpt_path` 等）正确。

### 4.2 训练 (Training)

训练脚本会自动读取配置，检查是否有中断的训练并询问是否恢复。每个 epoch 都会保存模型权重。

```bash
python train.py
```

*   **输出**: 日志和模型权重保存在 `logs/` 目录下。

### 4.3 预测 (Prediction)

`predict.py` 支持两种模式：单张图片预测和测试集评估。

**1. 单张图片预测**
预测单张图片，将检测框画在图上并显示置信度，保存到指定目录。

```bash
# 使用命令行参数
python predict.py --config ./config.yaml --image ./dataset/images/test/sample.jpg --out ./predictions/single

# 或者仅使用 config.yaml 中的配置
python predict.py --config ./config.yaml
```

**2. 测试集评估**
评估整个测试集，计算 Precision, Recall, mIoU 等指标。

```bash
python predict.py --config ./config.yaml --test
```

### 4.4 实时检测 (Webcam)

使用电脑摄像头进行实时二维码检测。

```bash
python webcam_detect.py --config ./config.yaml
```

**参数说明**:
*   `--index`: 摄像头索引（默认 0）。例如使用第二个摄像头：`--index 1`。
*   `--record`: 启动时直接开始录制视频。

**按键操作**:
*   **`q`**: 退出程序。
*   **`r`**: 开始/停止 录制视频。录制的视频将保存在 `config.yaml` 中 `live_save_dir` 指定的目录。

## 5. 目录结构说明

```
yolo1_QR/
├── config.yaml         # 项目配置文件
├── train.py            # 训练脚本
├── predict.py          # 预测与评估脚本
├── webcam_detect.py    # 摄像头实时检测脚本
├── models/             # 模型定义
│   ├── yolov1QR.py     # YOLOv1 网络结构
│   ├── loss.py         # 损失函数
│   └── ...
├── utils/              # 工具函数
├── dataset/            # 数据集目录
└── logs/               # 训练日志与权重保存目录
```
