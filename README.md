# 🎙️ DSP_test - 数字信号处理语音识别实验项目

本项目是一个综合性的数字信号处理（DSP）实验平台，涵盖语音信号的时域分析、频域分析和说话人识别三大核心实验。项目采用 Python 实现，并提供 Streamlit Web 界面进行交互式演示。

---

## 📑 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [实验内容](#实验内容)
  - [实验1：时域分析](#实验1时域分析)
  - [实验2：频域分析](#实验2频域分析)
  - [实验3：说话人识别](#实验3说话人识别)
- [使用方法](#使用方法)
- [算法原理](#算法原理)
- [实验结果](#实验结果)
- [参考文献](#参考文献)

---

## 项目概述

本项目实现了以下核心功能：

| 实验模块 | 核心技术 | 主要任务 |
|---------|---------|---------|
| 实验1 - 时域分析 | 短时能量 (STE)、过零率 (ZCR) | 语音端点检测、数字语音分类 |
| 实验2 - 频域分析 | MFCC、DTW 动态时间规整 | 孤立词识别 |
| 实验3 - 说话人识别 | GMM 高斯混合模型 | 独立于文本的说话人身份识别 |

---

## 项目结构

```
DSP_test/
├── dsp_app.py                      # Streamlit Web 应用主程序
├── README.md                       # 项目说明文档
│
├── exp1_programs/                  # 实验1：时域分析
│   ├── exp1_get_wav.py            # WAV 文件手动解析器
│   ├── exp1_segment_wav.py        # 端点检测与语音切分
│   └── exp1_compare_classifiers.py # 多分类器比较
│
├── exp2_programs/                  # 实验2：频域分析
│   ├── exp2_features.py           # MFCC 特征提取
│   ├── exp2_utils.py              # 工具函数（预加重、路径管理）
│   ├── dtw_algo.py                # DTW 算法实现
│   └── exp2_main.py               # 实验主程序
│
├── exp3_programs/                  # 实验3：说话人识别
│   └── exp3_speaker_id.py         # GMM 说话人识别系统
│
├── raw_wav/                        # 原始录音文件
│   ├── sfm/                       # 说话人1录音
│   └── wzh/                       # 说话人2录音
│
├── dataset/                        # 实验1&2 数据集（切分后的数字语音）
│   ├── 0/                         # 数字 "0" 的语音样本
│   ├── 1/                         # 数字 "1" 的语音样本
│   └── ...                        # 数字 2-9
│
├── dataset_exp3/                   # 实验3 说话人数据集
│   ├── Speaker_1_sfm/             # 说话人1（训练+测试文件）
│   ├── Speaker_2_cag/             # 说话人2
│   ├── Speaker_3_wzh/             # 说话人3
│   └── Speaker_6_cq/              # 说话人4
│
├── exp1_plots/                     # 实验1 输出图表
│   ├── segmentation/              # 端点检测可视化
│   └── confusion_matrices/        # 分类器混淆矩阵
│
├── exp2_plots/                     # 实验2 输出图表
│
├── exp3_plots/                     # 实验3 输出图表
│
└── test_programs/                  # 测试脚本（开发调试用）
```

---

## 环境配置

### 依赖安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install numpy scipy matplotlib seaborn pandas
pip install scikit-learn librosa joblib
pip install streamlit  # 可选：用于 Web 界面
```

### 依赖列表

| 库名称 | 用途 |
|--------|------|
| `numpy` | 数值计算 |
| `scipy` | 科学计算 |
| `matplotlib` | 图表绘制 |
| `seaborn` | 统计可视化 |
| `pandas` | 数据处理 |
| `scikit-learn` | 机器学习算法 |
| `librosa` | 音频处理与特征提取 |
| `joblib` | 模型持久化 |
| `streamlit` | Web 界面（可选） |

### Python 版本

推荐使用 **Python 3.8+**

---

## 实验内容

### 实验1：时域分析

**目标**：基于时域特征实现语音端点检测和数字语音分类。

#### 主要功能

1. **WAV 文件解析** (`exp1_get_wav.py`)
   - 手动解析 WAV 文件头（RIFF 格式）
   - 支持 8/16/32 位深度
   - 自动转换为单声道并归一化

2. **端点检测与切分** (`exp1_segment_wav.py`)
   - 基于短时能量 (STE) 的语音活动检测
   - 基于过零率 (ZCR) 的辅助判断
   - 自适应阈值策略
   - 自动切分连续数字语音

3. **多分类器比较** (`exp1_compare_classifiers.py`)
   - 朴素贝叶斯 (Naive Bayes)
   - Fisher 线性判别分析 (LDA)
   - 决策树 (Decision Tree)
   - K近邻 (KNN)
   - 支持向量机 (SVM)

#### 运行方式

```bash
# 步骤1：切分原始语音
cd exp1_programs
python exp1_segment_wav.py

# 步骤2：训练并比较分类器
python exp1_compare_classifiers.py
```

---

### 实验2：频域分析

**目标**：基于 MFCC 特征和 DTW 算法实现孤立词识别。

#### 主要功能

1. **MFCC 特征提取** (`exp2_features.py`)
   - 预加重滤波
   - Mel 滤波器组
   - 离散余弦变换 (DCT)
   - 倒谱均值归一化 (CMN)

2. **DTW 动态时间规整** (`dtw_algo.py`)
   - 欧氏距离矩阵计算
   - 动态规划最优路径
   - 最优路径可视化

3. **综合评估** (`exp2_main.py`)
   - MFCC 统计特征 + 传统分类器
   - DTW 序列匹配分类器
   - 原理图自动生成（Mel 滤波器、MFCC 热力图、DTW 路径）

#### 运行方式

```bash
cd exp2_programs
python exp2_main.py
```

#### 输出图表

- `principle_mel_filters.png` - Mel 滤波器组可视化
- `principle_mfcc_heatmap.png` - MFCC 特征热力图
- `principle_dtw_match.png` - 同类 DTW 路径
- `principle_dtw_mismatch.png` - 异类 DTW 路径
- `cm_*.png` - 各分类器混淆矩阵

---

### 实验3：说话人识别

**目标**：基于 GMM 实现独立于文本的说话人识别系统。

#### 主要功能

1. **音频预处理** (`exp3_speaker_id.py`)
   - 重采样至 16kHz
   - 语音活动检测 (VAD)
   - 静音切除

2. **特征提取**
   - 20维 MFCC
   - 一阶差分 (Delta MFCC)
   - 组合为 40 维特征向量

3. **GMM 声纹建模**
   - 每位说话人训练一个 16 分量 GMM
   - 对角协方差矩阵
   - 基于对数似然度的身份判决

#### 数据集格式

```
dataset_exp3/
├── Speaker_1_xxx/
│   ├── train_01.wav    # 训练文件（文件名包含 'train'）
│   ├── train_02.wav
│   ├── test_01.wav     # 测试文件（文件名包含 'test'）
│   └── test_02.wav
└── Speaker_2_xxx/
    └── ...
```

#### 运行方式

```bash
cd exp3_programs
python exp3_speaker_id.py
```

---

## 使用方法

### 方式一：命令行运行

分别进入各实验目录运行对应脚本（见上文各实验的运行方式）。

### 方式二：Streamlit Web 界面

```bash
# 在项目根目录运行
streamlit run dsp_app.py
```

Web 界面功能：
- 📂 原始数据浏览与试听
- ▶️ 一键运行实验脚本
- 📊 实验结果可视化展示

---

## 算法原理

### 1. 短时能量 (Short-Time Energy, STE)

$$E_n = \sum_{m=0}^{N-1} [x(n+m) \cdot w(m)]^2$$

其中 $x(n)$ 为语音信号，$w(m)$ 为窗函数，$N$ 为帧长。

### 2. 过零率 (Zero Crossing Rate, ZCR)

$$Z_n = \frac{1}{2N} \sum_{m=0}^{N-1} |\text{sgn}[x(n+m+1)] - \text{sgn}[x(n+m)]|$$

### 3. MFCC 提取流程

```
原始信号 → 预加重 → 分帧加窗 → FFT → Mel滤波器组 → 对数 → DCT → MFCC
```

### 4. DTW 递推公式

$$D(i,j) = d(i,j) + \min\{D(i-1,j), D(i,j-1), D(i-1,j-1)\}$$

其中 $d(i,j)$ 为第 $i$ 帧与第 $j$ 帧的欧氏距离。

### 5. GMM 似然度计算

$$\log p(\mathbf{X}|\lambda) = \sum_{t=1}^{T} \log \left[ \sum_{k=1}^{K} w_k \cdot \mathcal{N}(\mathbf{x}_t | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]$$

---

## 实验结果

### 实验1 分类器性能对比（示例）

| 算法 | 准确率 |
|------|--------|
| SVM (RBF) | 95.2% |
| KNN (k=5) | 93.8% |
| Fisher LDA | 91.5% |
| Decision Tree | 88.7% |
| Naive Bayes | 85.3% |

### 实验2 孤立词识别（示例）

| 方法 | 准确率 |
|------|--------|
| DTW Matching | 96.7% |
| SVM (MFCC Stats) | 94.1% |
| KNN (MFCC Stats) | 92.3% |

### 实验3 说话人识别（示例）

- 4 位说话人
- 整体识别准确率：约 95%+

---

## 文件说明

| 文件 | 描述 |
|------|------|
| `exp1_get_wav.py` | WAV 文件手动解析，不依赖第三方库读取 WAV |
| `exp1_segment_wav.py` | 自适应端点检测算法，支持连续数字切分 |
| `exp1_compare_classifiers.py` | 5 种分类器的训练与评估 |
| `exp2_features.py` | MFCC 特征提取，支持原始帧和统计特征 |
| `dtw_algo.py` | DTW 算法核心实现与可视化 |
| `exp2_main.py` | 实验2主程序，生成原理图与评估报告 |
| `exp3_speaker_id.py` | GMM 说话人识别完整流程 |
| `dsp_app.py` | Streamlit Web 应用界面 |

---

## 注意事项

1. **路径配置**：各脚本中的 `BASE_PATH` 需根据实际环境修改
2. **数据准备**：
   - 实验1&2：需先运行 `exp1_segment_wav.py` 生成 `dataset/` 数据
   - 实验3：需在 `dataset_exp3/` 中准备好训练和测试文件
3. **文件命名**：实验3 的音频文件需包含 `train` 或 `test` 关键字
4. **音频格式**：支持标准 WAV 格式（PCM 编码）

---

## 参考文献

1. 数字信号处理实验指导书
2. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*.
3. Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*.
4. Reynolds, D. A., & Rose, R. C. (1995). Robust text-independent speaker identification using Gaussian mixture speaker models. *IEEE Transactions on Speech and Audio Processing*.

---

## 作者

自动化2301五位同学

---

## License

本项目仅供学习交流使用。
