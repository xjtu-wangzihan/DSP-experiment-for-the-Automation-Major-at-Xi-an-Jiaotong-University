import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from exp1_get_wav import read_wav_manual

# ================= 配置路径 =================
BASE_PATH = r'C:\Users\12427\Desktop\数字信号处理\DSP_test'
DATASET_DIR = os.path.join(BASE_PATH, 'dataset')
PLOT_DIR = os.path.join(BASE_PATH, 'exp1_plots', 'confusion_matrices')

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# ================= 特征提取 =================
def extract_features_for_classification(file_path):
    """
    对单个切片文件提取统计特征：
    基于短时能量(STE)和过零率(ZCR)的 Mean, Std, Max
    """
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None
    
    # 简单的分帧逻辑
    frame_len = int(0.025 * sr)
    step_len = int(0.010 * sr)
    
    # 补齐
    if len(signal) < frame_len: return None
    num_frames = (len(signal) - frame_len) // step_len + 1
    
    # 计算 STE 和 ZCR
    ste_list = []
    zcr_list = []
    
    for i in range(num_frames):
        start = i * step_len
        frame = signal[start : start + frame_len]
        
        # STE
        ste = np.sum(frame ** 2)
        ste_list.append(ste)
        
        # ZCR
        signs = np.sign(frame)
        signs[signs==0] = 1
        zcr = 0.5 * np.sum(np.abs(signs[1:] - signs[:-1]))
        zcr_list.append(zcr)
        
    ste_arr = np.array(ste_list)
    zcr_arr = np.array(zcr_list)
    
    # 构造特征向量 (6维 + 时长 = 7维)
    features = [
        np.mean(ste_arr), np.std(ste_arr), np.max(ste_arr),
        np.mean(zcr_arr), np.std(zcr_arr), np.max(zcr_arr),
        len(signal)/sr
    ]
    return np.array(features)

def load_data():
    X, y = [], []
    print("正在从 dataset 文件夹加载并提取特征...")
    
    total_files = 0
    # 遍历 0-9 文件夹
    for label in range(10):
        folder = os.path.join(DATASET_DIR, str(label))
        if not os.path.exists(folder): continue
        
        files = glob.glob(os.path.join(folder, "*.wav"))
        for f in files:
            feat = extract_features_for_classification(f)
            if feat is not None:
                X.append(feat)
                y.append(label)
                total_files += 1
                
    print(f"共加载 {total_files} 个样本。")
    return np.array(X), np.array(y)

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 准备数据
    X, y = load_data()
    if len(X) == 0:
        print("错误：数据集中没有有效文件，请先运行 exp1_segment_wav.py")
        exit()

    # 数据划分与标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 2. 定义分类器 (根据手册要求选取) [cite: 129]
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Fisher LDA": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf')
    }
    
    results = []
    
    print("\n开始训练与评估...")
    for name, clf in classifiers.items():
        # 训练
        clf.fit(X_train, y_train)
        # 预测
        y_pred = clf.predict(X_test)
        # 精度
        acc = accuracy_score(y_test, y_pred)
        results.append({"Algorithm": name, "Accuracy": acc})
        
        print(f"[{name}] Accuracy: {acc:.4f}")
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {name}\nAcc: {acc:.2%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, f"cm_{name.replace(' ','_')}.png")
        plt.savefig(save_path)
        plt.close()
    
    # 3. 汇总输出
    print("\n" + "="*30)
    print("最终结果汇总")
    print("="*30)
    df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(df)
    print(f"\n所有混淆矩阵图片已保存至: {PLOT_DIR}")