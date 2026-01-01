import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# [cite_start]引入 sklearn 所有分类器 (实验指导书要求 [cite: 129])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 引入自定义模块 (假设你已经保存了 exp2_features.py 和 dtw_algo.py)
from exp2_features import extract_mfcc_stats, extract_mfcc_sequence
from dtw_algo import DTWClassifier

def load_data(dataset_dir):
    # 轨道A数据 (定长统计特征)
    X_stats, y = [], []
    # 轨道B数据 (原始序列特征)
    X_seqs = [] 
    
    print("正在提取双轨特征 (MFCC Stats & Sequences)...")
    for label in range(10):
        folder_path = os.path.join(dataset_dir, str(label))
        # 确保按文件名排序读取，保证对应关系
        wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
        
        for wav_file in wav_files:
            # 1. 提取统计特征 (给 SVM/KNN 等用)
            stat_feat = extract_mfcc_stats(wav_file)
            # 2. 提取序列特征 (给 DTW 用)
            seq_feat = extract_mfcc_sequence(wav_file)
            
            if stat_feat is not None and seq_feat is not None:
                X_stats.append(stat_feat)
                X_seqs.append(seq_feat)
                y.append(label)
                
    return np.array(X_stats), X_seqs, np.array(y)

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=[str(i) for i in range(10)], 
                yticklabels=[str(i) for i in range(10)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print("错误: dataset 目录不存在，请检查路径。")
        exit()

    # ===========================
    # 1. 数据准备
    # ===========================
    X_stats, X_seqs, y = load_data(dataset_dir)
    
    # 划分数据集 (固定随机种子 random_state=42 以确保实验可复现)
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    
    # 准备统计流数据 (Stat) - 需要标准化
    scaler = StandardScaler()
    X_stats_scaled = scaler.fit_transform(X_stats)
    X_train_stat = X_stats_scaled[idx_train]
    X_test_stat = X_stats_scaled[idx_test]
    
    # 准备序列流数据 (Seq) - DTW 不需要标准化或需单独处理，这里直接使用
    X_train_seq = [X_seqs[i] for i in idx_train]
    X_test_seq = [X_seqs[i] for i in idx_test]
    
    y_train = y[idx_train]
    y_test = y[idx_test]
    
    results = []
    
    # ===========================
    # 2. 第一阶段: 统计特征分类器对比
    # ===========================
    print("\n========== 第一阶段: 基于 MFCC 统计特征的分类 ==========")
    # [cite_start]包含手册要求的5种分类器 [cite: 129]
    classifiers = {
        "Naïve Bayesian": GaussianNB(),
        "Fisher LDA": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel='rbf'),
        "KNN (k=1)": KNeighborsClassifier(n_neighbors=1)
    }
    
    for name, clf in classifiers.items():
        clf.fit(X_train_stat, y_train)
        pred = clf.predict(X_test_stat)
        acc = accuracy_score(y_test, pred)
        results.append({"Algorithm": name, "Type": "Statistical", "Accuracy": acc})
        print(f"  [{name}] 准确率: {acc:.2%}")
        plot_cm(y_test, pred, f'CM - {name}', f'cm_exp2_{name.replace(" ","_")}.png')

    # ===========================
    # 3. 第二阶段: DTW 序列匹配
    # ===========================
    print("\n========== 第二阶段: 基于 DTW 的序列匹配 ==========")
    # [cite_start]对应手册中关于 DTW 处理不等长向量的要求 [cite: 158-161]
    dtw_clf = DTWClassifier()
    dtw_clf.fit(X_train_seq, y_train)
    dtw_pred = dtw_clf.predict(X_test_seq)
    dtw_acc = accuracy_score(y_test, dtw_pred)
    
    results.append({"Algorithm": "DTW Matching", "Type": "Sequence (DTW)", "Accuracy": dtw_acc})
    print(f"  [DTW Matching] 准确率: {dtw_acc:.2%}")
    plot_cm(y_test, dtw_pred, 'CM - DTW', 'cm_exp2_DTW.png')

    # ===========================
    # 4. 最终结果汇总与并列排名处理
    # ===========================
    print("\n" + "="*60)
    print("         实验 2: 频域识别算法全面评估排行榜")
    print("="*60)
    
    # 创建 DataFrame 并按准确率降序排列
    df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(df.to_string(index=False))
    print("-" * 60)
    
    # --- 修改点：处理并列第一的情况 ---
    # 1. 获取最高准确率数值
    max_acc = df['Accuracy'].max()
    
    # 2. 筛选出所有等于最高准确率的行
    top_performers = df[df['Accuracy'] == max_acc]
    
    # 3. 获取这些算法的名称列表
    winner_names = top_performers['Algorithm'].tolist()
    
    print(f"\n本次实验的最佳表现准确率为: {max_acc:.2%}")
    
    if len(winner_names) > 1:
        print(f"结论: 共有 {len(winner_names)} 个算法并列第一，展现了最优性能:")
        for name in winner_names:
            print(f"  ★ {name}")
        print("这表明 MFCC 特征具有极高的鲁棒性，使得多种分类器均能达到最佳效果。")
    else:
        print(f"结论: 表现最好的算法是 [{winner_names[0]}]。")
        
    print("="*60)