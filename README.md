# 基于 MobileNetV2 迁移学习的猫狗图像二分类

面向 CPU 设备的轻量化图像二分类项目。项目基于 TensorFlow / Keras，以 ImageNet 预训练的 MobileNetV2 作为冻结主干，通过“特征缓存 + 轻量分类头”实现迁移学习，在没有独立显卡的笔记本上完成 5 轮训练仅需约 13 秒。

本项目来源于深度学习课程期末实验，代码已针对低功耗 CPU 环境完成性能优化与数据清洗，并配套完整的中文说明文档。

## 目录

- [项目简介](#项目简介)
- [效果预览](#效果预览)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [参数配置](#参数配置)
- [实验结果](#实验结果)
- [性能优化](#性能优化)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [文档与报告](#文档与报告)

## 项目简介

猫狗图像二分类是计算机视觉的经典入门任务。本项目使用 Kaggle 猫狗数据集，以 MobileNetV2 作为特征提取器，在其后接一个轻量分类头（全局平均池化 → 全连接层 → Dropout → Softmax），实现“猫 / 狗”二分类。

项目的重点不只是完成分类任务，还包括：

1. **迁移学习**：复用 ImageNet 预训练权重，在小规模数据上快速收敛；
2. **CPU 性能优化**：冻结主干后只计算一次特征，避免每轮重复前向；
3. **数据质量治理**：识别并剔除数据集中被错误标注的样本；
4. **完整实验流程**：数据加载、训练、评估、可视化一体化。

## 效果预览

### 训练样本

![训练集样本](docs/images/01-training-samples.png)

_图 1 训练集猫狗样本可视化_

### 训练过程

![训练损失与准确率曲线](docs/images/02-training-curves.png)

_图 2 训练损失与准确率变化曲线（5 轮，训练准确率 98.39%）_

### 测试集预测

![测试集预测结果](docs/images/03-test-predictions.png)

_图 3 测试集 8 张图片的预测结果与置信度_

## 核心特性

- **高精度**：5 轮训练后训练准确率 **98.39%**，测试集 8/8 全部预测正确。
- **低门槛**：无独立显卡也能训练，5 轮训练约 **12.7 秒**。
- **特征缓存**：冻结主干只前向计算一次，之后每轮仅训练约 8 万参数的分类头。
- **数据清洗**：自动过滤 `dog/` 目录中约 43% 被误标为狗的猫图，避免标签噪声。
- **可复现**：固定随机种子，结果稳定可复现。
- **中文可视化**：自动生成训练样本、训练曲线、预测结果与性能对比图。

## 技术架构

![技术架构与数据流](docs/images/04-architecture.png)

_图 4 系统技术架构与数据流_

整体流程分为四个阶段：

1. **数据准备与清洗**：扫描 `cat/`、`dog/` 目录，仅保留 `cat.*` / `dog.*` 原始样本，按类别抽取训练集与测试集。
2. **图像预处理**：JPEG 快速解码，缩放至 128×128，像素归一化到 [0, 1]。
3. **特征提取**：MobileNetV2（`include_top=False`、ImageNet 权重、冻结）输出 4×4×1280 特征图，经全局平均池化得到 1280 维向量，并缓存为 `992×1280` 的特征矩阵。
4. **分类头训练与推理**：训练 `Dense(64) → Dropout(0.3) → Dense(2, Softmax)`，训练完成后与主干组合成完整模型用于预测。

## 环境要求

| 项目       | 要求                                                                    |
| ---------- | ----------------------------------------------------------------------- |
| 操作系统   | Windows 10/11（已验证），Linux/macOS 同理                               |
| Python     | 3.11（已在 3.11.9 验证）；其他版本请以 TensorFlow 2.20 官方支持列表为准 |
| TensorFlow | 2.20.0（CPU 版即可）                                                    |
| 其他依赖   | NumPy、Matplotlib、Pillow                                               |
| 硬件       | CPU 即可，内存建议 8 GB 以上；无 GPU 也可运行                           |
| 已验证硬件 | Intel i5-8365U（4 核 8 线程）/ 16 GB / Intel UHD 620                    |

## 快速开始

### 1. 准备数据集

将 Kaggle 猫狗数据集按以下结构放置：

```text
train/
├── cat/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   └── ...
└── dog/
    ├── dog.0.jpg
    ├── dog.1.jpg
    └── ...
```

代码默认读取 `G:\深度学习与开发\Jupyter Notebook\train`。如路径不同，请修改 [dogs-vs-cats.py](dogs-vs-cats.py) 顶部的 `DATA_ROOT`。

> 注意：代码只读取 `cat.*` 与 `dog.*` 命名的图片。若使用其他数据集，请同步修改 `list_images()` 中的文件名前缀。

### 2. 创建虚拟环境并安装依赖

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install tensorflow==2.20.0 matplotlib pillow
```

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.20.0 matplotlib pillow
```

### 3. 运行

```bash
python dogs-vs-cats.py
```

运行后将输出模型结构、训练日志、测试准确率，并弹出 4 张可视化图表。

## 参数配置

所有超参数集中在 [dogs-vs-cats.py](dogs-vs-cats.py) 顶部：

| 参数                | 默认值         | 说明                      |
| ------------------- | -------------- | ------------------------- |
| `DATA_ROOT`         | `G:\...\train` | 数据集根目录              |
| `IMG_SIZE`          | 128            | 输入图像边长              |
| `BATCH_SIZE`        | 4              | 分类头训练批大小          |
| `INFER_BATCH_SIZE`  | 32             | 特征提取 / 测试推理批大小 |
| `TRAIN_EACH_MAX`    | 500            | 每类最多使用的训练样本数  |
| `TEST_EACH_NUM`     | 4              | 每类抽取的测试样本数      |
| `EPOCHS`            | 5              | 训练轮数                  |
| `USE_FEATURE_CACHE` | True           | 是否启用特征缓存加速      |
| `RANDOM_SEED`       | 42             | 随机种子                  |

## 实验结果

### 训练指标

| 轮次 | 损失 Loss | 准确率 Accuracy |
| ---- | --------- | --------------- |
| 1    | 0.2088    | 92.44%          |
| 2    | 0.0895    | 96.98%          |
| 3    | 0.0539    | 98.39%          |
| 4    | 0.0460    | 98.19%          |
| 5    | 0.0357    | 98.39%          |

### 优化前后对比

![性能对比](docs/images/05-performance.png)

_图 5 同一台电脑上的运行耗时对比_

| 指标         | 优化前                 | 优化后                        |
| ------------ | ---------------------- | ----------------------------- |
| 5 轮训练耗时 | 约 67 秒               | 12.7 秒                       |
| 全流程耗时   | 约 75 秒               | 25.9 秒（含 TensorFlow 导入） |
| 训练准确率   | 约 84%                 | 98.39%                        |
| 测试集结果   | 5/8                    | 8/8                           |
| 数据标签     | `dog/` 混入约 43% 猫图 | 仅使用 `cat.*` / `dog.*`      |

> 测试集仅 8 张图片，100% 准确率不具备统计显著性，建议将 `TEST_EACH_NUM` 调整为 50–100 后重新评估。

## 性能优化

| 优化项   | 做法                                 | 效果                          |
| -------- | ------------------------------------ | ----------------------------- |
| 特征缓存 | 冻结主干只前向一次，缓存 1280 维特征 | 训练耗时降低约 5 倍           |
| 线程控制 | 固定 4 个计算线程（物理核数）        | 减少 15 W CPU 过热降频        |
| 快速解码 | JPEG `INTEGER_FAST` + 2 路并行解码   | 首次特征提取约 8 秒（含解码） |
| 全局打乱 | 使用 Keras 全局 shuffle 训练分类头   | 修复同类样本扎堆，收敛更稳    |
| 数据清洗 | 仅保留 `cat.*` / `dog.*`             | 训练准确率提升至 98.39%       |
| 批量推理 | `INFER_BATCH_SIZE=32`                | 测试推理更快                  |
| 单次绘图 | 所有图最后只调用一次 `plt.show()`    | 避免多次阻塞                  |

## 项目结构

```text
猫狗大战/
├── dogs-vs-cats.py                          # 主程序（数据、模型、训练、评估、可视化）
├── docs/
│   ├── 项目说明文档.md                       # 完整项目说明文档
│   ├── PUBLISH.md                           # GitHub / Gitee / CSDN 发布说明
│   └── images/                              # 文档配图
│       ├── 01-training-samples.png
│       ├── 02-training-curves.png
│       ├── 03-test-predictions.png
│       ├── 04-architecture.png
│       └── 05-performance.png
├── 基于MobileNetV2迁移学习的猫狗图像二分类实验报告.pdf
├── Figure 2026-06-19 041626.png             # 优化前训练曲线（存档）
├── Figure 2026-06-19 041648.png             # 优化前预测结果（存档）
├── 深度学习实验报告模版.docx
└── 深度学习评分标准.pdf
```

> `.venv/` 为本地虚拟环境，体积约 1.6 GB，请勿上传到代码托管平台。

## 常见问题

### 在 Spyder 中重复运行时报错 `Intra op parallelism cannot be modified after initialization`

TensorFlow 的线程池只能在运行时初始化前配置。脚本已捕获该异常并沿用当前配置；如需应用 4 线程设置，请在 Spyder 中重启内核（`Ctrl+.`）后再运行。

### 报错 `FileNotFoundError: 训练目录缺少 cat 或 dog 子目录`

请检查 `DATA_ROOT` 是否正确，并确认目录下存在 `cat/` 与 `dog/` 两个子文件夹。

### 图表中文显示为方块

请确认系统安装了 SimHei（黑体）。脚本默认使用 `SimHei`，也可在文件顶部替换为其他中文字体。

### 训练速度忽快忽慢

本机为 15 W 低功耗处理器，长时间训练会触发温度墙降频。建议插电运行、使用“高性能”电源计划，并保持散热口通畅。

更多问题与解决方案见 [完整项目说明文档](docs/项目说明文档.md)。

## 文档与报告

- [项目说明文档](docs/项目说明文档.md)：项目介绍、功能演示、技术架构、使用指南、性能优化、实验分析与常见问题
- [多平台发布说明](docs/PUBLISH.md)：GitHub、Gitee、CSDN 的发布要点与图片处理方式
- [实验报告 PDF](基于MobileNetV2迁移学习的猫狗图像二分类实验报告.pdf)：课程实验报告（优化前版本）

## 数据与版权说明

- 数据集：Kaggle Dogs vs Cats，版权归数据集原作者所有，本项目仅用于课程学习与研究。
- 预训练权重：ImageNet 上训练的 MobileNetV2，通过 Keras Applications 自动下载。
- 本项目代码仅用于学习与交流，请勿用于商业用途。

## 参考

1. Sandler M., Howard A., Zhu M., et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, 2018.
2. TensorFlow 2.20 官方文档：https://www.tensorflow.org/
3. Keras Applications：https://keras.io/api/applications/
4. Kaggle Dogs vs. Cats：https://www.kaggle.com/c/dogs-vs-cats
