import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 引入 sklearn 的各种分类器
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 朴素贝叶斯 (Naïve Bayes)
from sklearn.naive_bayes import GaussianNB
# 2. Fisher 线性判别 (LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 3. 决策树 (Decision Tree)
from sklearn.tree import DecisionTreeClassifier
# 4. KNN
from sklearn.neighbors import KNeighborsClassifier
# 5. SVM
from sklearn.svm import SVC

# 复用之前的读取函数
from get_wav import read_wav_manual

# ===========================
# 特征提取部分 (保持不变)
# ===========================
def to_mono(signal):
    if signal.ndim > 1:
        return signal[:, 0]
    return signal

def enframe(signal, frame_len, step_len):
    signal_len = len(signal)
    if signal_len <= frame_len:
        return np.array([signal])
    num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / step_len))
    pad_len = int((num_frames - 1) * step_len + frame_len)
    zeros = np.zeros((pad_len - signal_len,))
    pad_signal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * step_len, step_len), (frame_len, 1)).T
    return pad_signal[indices.astype(np.int32)]

def extract_features(file_path):
    # 1. 读取
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None
    signal = to_mono(signal)
    signal = signal - np.mean(signal)
    
    # 2. 分帧
    frame_len = int(0.025 * sr)
    step_len = int(0.010 * sr)
    frames = enframe(signal, frame_len, step_len)
    
    # 3. 特征计算
    ste = np.sum(frames ** 2, axis=1) # 短时能量
    signs = np.sign(frames)
    diffs = np.abs(signs[:, 1:] - signs[:, :-1])
    zcr = 0.5 * np.sum(diffs, axis=1) # 过零率
    
    # 4. 统计特征
    features = [
        np.mean(ste), np.std(ste), np.max(ste),
        np.mean(zcr), np.std(zcr), np.max(zcr),
        len(signal) / sr
    ]
    return np.array(features)

def load_dataset(dataset_dir):
    X, y = [], []
    print("正在加载数据集...")
    for label in range(10):
        folder_path = os.path.join(dataset_dir, str(label))
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        for wav_file in wav_files:
            feat = extract_features(wav_file)
            if feat is not None:
                X.append(feat)
                y.append(label)
    return np.array(X), np.array(y)

# ===========================
# 绘图工具
# ===========================
def save_confusion_matrix(y_true, y_pred, title, filename):
    """保存混淆矩阵图片，不弹窗"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(10)], 
                yticklabels=[str(i) for i in range(10)])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # 关闭画布，防止内存泄露
    print(f"  -> 已保存: {filename}")

# ===========================
# 主程序：五大算法对决
# ===========================
if __name__ == "__main__":
    dataset_dir = "dataset"
    
    # 1. 准备数据
    X, y = load_dataset(dataset_dir)
    if len(X) == 0:
        print("错误：未找到数据。")
        exit()
        
    # 标准化 (对于SVM, KNN, LDA都很重要)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"样本总数: {len(X)} (训练集: {len(X_train)}, 测试集: {len(X_test)})")
    print("-" * 50)

    # 2. 定义分类器库 
    # 这里包含了实验指导书要求的所有算法
    classifiers = {
        # 朴素贝叶斯: 假设特征之间相互独立，计算简单
        "Naive Bayes": GaussianNB(),
        
        # Fisher 线性判别: 在sklearn中即为 LDA (Linear Discriminant Analysis)
        "Fisher LDA": LinearDiscriminantAnalysis(),
        
        # 决策树: 基于树状结构进行规则划分
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        
        # KNN: 之前的基准
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        
        # SVM: 之前的冠军
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # 存储结果用于最后排名
    results = []

    # 3. 循环训练并评估
    for name, clf in classifiers.items():
        print(f"\n>>> 正在测试: {name} <<<")
        
        # 训练
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 评估
        acc = accuracy_score(y_test, y_pred)
        results.append({"Algorithm": name, "Accuracy": acc})
        
        print(f"  准确率: {acc:.2%}")
        
        # 保存每个算法的混淆矩阵，方便报告中使用
        # 文件名例如: confusion_matrix_Decision_Tree.png
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        save_confusion_matrix(y_test, y_pred, 
                             f'Confusion Matrix - {name}', 
                             f'confusion_matrix_{safe_name}.png')

    # 4. 最终排名汇总 
    print("\n" + "="*30)
    print("     算法性能排行榜")
    print("="*30)
    
    # 转为 DataFrame 并排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    print(df_results)
    print("="*30)
    
    best_algo = df_results.iloc[0]['Algorithm']
    print(f"\n结论: 在当前时域特征下，表现最好的算法是 [{best_algo}]。")
    print("请将生成的 5 张 confusion_matrix 图片插入实验报告中进行对比分析。")