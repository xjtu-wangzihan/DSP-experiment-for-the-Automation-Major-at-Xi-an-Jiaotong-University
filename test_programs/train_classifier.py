import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 复用你修正后的读取函数
from get_wav import read_wav_manual

def to_mono(signal):
    """确保使用单声道数据"""
    if signal.ndim > 1:
        return signal[:, 0] # 修正后读取到的是(N, 2)，直接取左声道
    return signal

def enframe(signal, frame_len, step_len):
    """分帧函数"""
    signal_len = len(signal)
    if signal_len <= frame_len:
        return np.array([signal]) # 信号太短则作为一帧
        
    num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / step_len))
    pad_len = int((num_frames - 1) * step_len + frame_len)
    zeros = np.zeros((pad_len - signal_len,))
    pad_signal = np.concatenate((signal, zeros))
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * step_len, step_len), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    return pad_signal[indices]

def extract_features(file_path):
    """
    [Step 4: 时域分析]
    计算短时能量和过零率，并提取统计特征。
    """
    # 1. 读取数据
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None
    
    signal = to_mono(signal)
    
    # 简单去噪/去直流
    signal = signal - np.mean(signal)
    
    # 2. 分帧 (25ms 帧长, 10ms 帧移)
    frame_len = int(0.025 * sr)
    step_len = int(0.010 * sr)
    frames = enframe(signal, frame_len, step_len)
    
    # 3. 计算短时能量 (Short-Time Energy) [cite: 114-116]
    # En = sum(x[m]^2)
    ste = np.sum(frames ** 2, axis=1)
    
    # 4. 计算过零率 (Zero-Crossing Rate) [cite: 117-118]
    # Zn = 0.5 * sum(|sgn[x[m]] - sgn[x[m-1]]|)
    # 这里用一种高效的向量化实现：
    # 比较每一帧中相邻元素的符号是否不同
    signs = np.sign(frames)
    # 每一行的相邻元素差值
    diffs = np.abs(signs[:, 1:] - signs[:, :-1])
    zcr = 0.5 * np.sum(diffs, axis=1)
    
    # 5. 构造特征向量 (Feature Vector)
    # 因为不同语音时长不同，帧数N不同。
    # 为了送入分类器，我们需要提取由于N无关的统计特征。
    # 选取：均值(Mean), 标准差(Std), 最大值(Max)
    features = [
        np.mean(ste), np.std(ste), np.max(ste),  # 能量特征
        np.mean(zcr), np.std(zcr), np.max(zcr),  # 过零率特征
        len(signal) / sr                         # 持续时长特征
    ]
    
    return np.array(features)

def load_dataset(dataset_dir):
    """遍历 dataset 文件夹加载所有数据"""
    X = [] # 特征矩阵
    y = [] # 标签列表
    
    print("正在提取特征...")
    # 遍历 0 到 9 的文件夹
    for label in range(10):
        folder_path = os.path.join(dataset_dir, str(label))
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        print(f"  类别 {label}: 发现 {len(wav_files)} 个样本")
        
        for wav_file in wav_files:
            feat = extract_features(wav_file)
            if feat is not None:
                X.append(feat)
                y.append(label)
                
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵 [Step 6: 实验对比及量化分析]"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵图已保存为 confusion_matrix.png")
    # plt.show()

# ===========================
# 主程序
# ===========================
if __name__ == "__main__":
    dataset_dir = "dataset"  # 你的数据目录
    
    # 1. 加载数据并提取特征
    X, y = load_dataset(dataset_dir)
    
    print(f"\n特征提取完成。样本总数: {len(X)}, 特征维数: {X.shape[1]}")
    
    if len(X) == 0:
        print("错误：未找到数据，请确保 'dataset' 文件夹内有音频文件。")
        exit()

    # 2. 数据标准化
    # 不同特征的量纲不同（能量很大，过零率很小），必须标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 划分训练集和测试集
    # 按照 8:2 或 7:3 划分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # ===========================
    # [Step 5: 语音识别分类器的实现]
    # 实验手册建议对比不同分类器 
    # 这里我们同时训练 KNN 和 SVM 两个模型进行对比
    # ===========================
    
    classifiers = {
        "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=3),
        "SVM (Support Vector Machine)": SVC(kernel='rbf', C=1.0, gamma='scale')
    }
    
    for name, clf in classifiers.items():
        print(f"\n--- 正在训练 {name} ---")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # [Step 6: 实验对比及量化分析]
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} 准确率: {acc:.2%}")
        print("详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
        
        # 如果是 SVM，我们画一下混淆矩阵作为示例
        if "SVM" in name:
            plot_confusion_matrix(y_test, y_pred, classes=[str(i) for i in range(10)])

    print("\n实验完成！")
    print("建议：")
    print("1. 检查 confusion_matrix.png,查看哪些数字容易混淆(例如 1 和 7,或者 6 和 9)。")
    print("2. 如果准确率不理想（低于80%），可能需要检查之前的端点检测是否切到了噪音，或者尝试采集更多样本。")