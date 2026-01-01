import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
# 分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 自定义模块
from exp2_utils import get_paths
from exp2_features import extract_mfcc, get_feature_stats
from dtw_algo import DTWClassifier, plot_dtw_path

# ================= 可视化功能 =================

def visualize_principles(dataset_dir, output_dir):
    """
    [可视化] 生成实验报告所需的原理图
    1. Mel 滤波器组
    2. MFCC 特征热力图
    3. DTW 匹配路径 (同类 vs 异类)
    """
    print(">>> 正在生成原理可视化图表 (保存至 exp2_plots)...")
    
    # 尝试寻找样本文件 (假设已有 dataset/1/xxx.wav)
    try:
        sample_digit_1 = glob.glob(os.path.join(dataset_dir, '1', '*.wav'))
        sample_digit_2 = glob.glob(os.path.join(dataset_dir, '2', '*.wav'))
        
        if len(sample_digit_1) < 2 or len(sample_digit_2) < 1:
            print("  警告: 数据集样本不足，跳过部分绘图。")
            return

        f1_a = sample_digit_1[0] # 数字1 样本A
        f1_b = sample_digit_1[1] # 数字1 样本B
        f2_a = sample_digit_2[0] # 数字2 样本A
    except IndexError:
        print("  警告: 找不到足够的音频文件进行可视化。")
        return

    # --- 1. 绘制 Mel 滤波器组 ---
    plt.figure(figsize=(10, 4))
    mels = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    for i in range(0, 40, 3): # 抽样绘制
        plt.plot(mels[i], label=f'Filter {i}')
    plt.title("Mel Filter Bank (Triangular Filters)")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "principle_mel_filters.png"))
    plt.close()
    
    # --- 2. 绘制 MFCC 热力图 ---
    mfcc_raw = extract_mfcc(f1_a, return_raw=True).T # 转置以便 librosa 绘图 (n_mfcc, Frames)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_raw, x_axis='time', sr=16000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"MFCC Heatmap (Digit '1')")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "principle_mfcc_heatmap.png"))
    plt.close()
    
    # --- 3. 绘制 DTW 路径 ---
    seq1 = extract_mfcc(f1_a) # (Frames, D)
    seq1_diff = extract_mfcc(f1_b)
    seq2 = extract_mfcc(f2_a)
    
    # Case A: 同类匹配 (1 vs 1)
    plot_dtw_path(seq1, seq1_diff, "Digit 1 vs Digit 1 (Same Class)", 
                  os.path.join(output_dir, "principle_dtw_match.png"))
    # Case B: 异类匹配 (1 vs 2)
    plot_dtw_path(seq1, seq2, "Digit 1 vs Digit 2 (Different Class)", 
                  os.path.join(output_dir, "principle_dtw_mismatch.png"))
    print("  原理图生成完毕。")

def load_data(dataset_dir):
    X_seq = []
    y = []
    print(">>> 正在加载数据集 (提取 MFCC)...")
    
    total_files = 0
    for label in range(10):
        folder = os.path.join(dataset_dir, str(label))
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        for f in files:
            mfcc = extract_mfcc(f)
            if mfcc is not None:
                X_seq.append(mfcc)
                y.append(label)
                total_files += 1
    print(f"  共加载 {total_files} 个样本。")
    return X_seq, np.array(y)

# ================= 主程序 =================

if __name__ == "__main__":
    # 获取路径配置
    paths = get_paths()
    
    # 检查数据集是否存在
    if not os.path.exists(paths['dataset']):
        print(f"严重错误: 未找到数据集目录: {paths['dataset']}")
        print("请先运行 exp1_segment_wav.py 生成数据。")
        exit(1)

    # 1. 生成原理可视化图
    visualize_principles(paths['dataset'], paths['plots'])

    # 2. 加载数据
    X_seqs, y = load_data(paths['dataset'])
    if len(X_seqs) == 0:
        print("错误: 数据集为空。")
        exit(1)
    
    # 3. 数据划分
    # 使用索引划分，保持序列特征和统计特征的对应关系
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    
    y_train = y[idx_train]
    y_test = y[idx_test]
    
    X_seq_train = [X_seqs[i] for i in idx_train]
    X_seq_test = [X_seqs[i] for i in idx_test]
    
    results = []

    # ================= 阶段一：统计特征 + 传统分类器 =================
    print("\n>>> [阶段一] MFCC 统计特征分类评估")
    
    # 提取统计特征 (Mean, Std)
    X_stats = np.array([get_feature_stats(x) for x in X_seqs])
    scaler = StandardScaler()
    X_stats_scaled = scaler.fit_transform(X_stats)
    
    X_stat_train = X_stats_scaled[idx_train]
    X_stat_test = X_stats_scaled[idx_test]
    
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Fisher LDA": LinearDiscriminantAnalysis(),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "SVM (RBF)": SVC(kernel='rbf'),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    for name, clf in classifiers.items():
        clf.fit(X_stat_train, y_train)
        pred = clf.predict(X_stat_test)
        acc = accuracy_score(y_test, pred)
        results.append({"Algorithm": name, "Features": "MFCC Stats", "Accuracy": acc})
        
        # 保存混淆矩阵
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(paths['plots'], f'cm_{name.replace(" ","_")}.png'))
        plt.close()
        print(f"  {name:<15}: {acc:.2%}")

    # ================= 阶段二：DTW 序列匹配 =================
    print("\n>>> [阶段二] DTW 序列匹配评估 (速度较慢，请稍候)")
    
    dtw_clf = DTWClassifier()
    dtw_clf.fit(X_seq_train, y_train)
    dtw_pred = dtw_clf.predict(X_seq_test)
    dtw_acc = accuracy_score(y_test, dtw_pred)
    
    results.append({"Algorithm": "DTW Matching", "Features": "MFCC Seq", "Accuracy": dtw_acc})
    
    # 保存 DTW 混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, dtw_pred), annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - DTW Matching')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(paths['plots'], 'cm_DTW_Matching.png'))
    plt.close()
    
    print(f"  {'DTW Matching':<15}: {dtw_acc:.2%}")

    # ================= 结果汇总 =================
    print("\n" + "="*50)
    print("实验2 算法性能排行榜")
    print("="*50)
    df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(df)
    
    print(f"\n所有生成的图表 (原理图 + 混淆矩阵) 已保存至:\n{paths['plots']}")