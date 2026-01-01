import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calc_dtw_cost(seq1, seq2):
    """
    [cite_start][实验2 原理4] DTW 动态时间规整算法 [cite: 158-161]
    seq1: 测试序列 (N, D)
    seq2: 模板序列 (M, D)
    """
    n, d1 = seq1.shape
    m, d2 = seq2.shape
    
    # 1. 计算欧氏距离矩阵 (利用广播加速)
    # diff: (N, M, D)
    diff = seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # 2. 初始化累积代价矩阵
    cost = np.full((n, m), np.inf)
    cost[0, 0] = dist_matrix[0, 0]
    
    # 3. 动态规划填表
    # D(i,j) = dist(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0: continue
            
            candidates = []
            if i > 0: candidates.append(cost[i-1, j])
            if j > 0: candidates.append(cost[i, j-1])
            if i > 0 and j > 0: candidates.append(cost[i-1, j-1])
            
            if candidates:
                cost[i, j] = dist_matrix[i, j] + min(candidates)
            
    # 归一化距离
    normalized_dist = cost[n-1, m-1] / (n + m)
    return normalized_dist, cost

def plot_dtw_path(seq1, seq2, title, save_path):
    """
    [可视化] 绘制 DTW 最优路径 (实验指导书图3复刻)
    """
    dist, cost_matrix = calc_dtw_cost(seq1, seq2)
    n, m = cost_matrix.shape
    
    # 回溯最优路径
    path = []
    i, j = n-1, m-1
    path.append((j, i))
    while i > 0 or j > 0:
        candidates = []
        if i > 0: candidates.append((cost_matrix[i-1, j], i-1, j)) # 上
        if j > 0: candidates.append((cost_matrix[i, j-1], i, j-1)) # 左
        if i > 0 and j > 0: candidates.append((cost_matrix[i-1, j-1], i-1, j-1)) # 对角
        
        # 贪心选择最小代价的前驱
        if not candidates: break
        _, next_i, next_j = min(candidates, key=lambda x: x[0])
        path.append((next_j, next_i))
        i, j = next_i, next_j
        
    path_x, path_y = zip(*path)
    
    plt.figure(figsize=(8, 6))
    
    # === [修复点] 删除 origin='lower'，改用 invert_yaxis ===
    sns.heatmap(cost_matrix.T, cmap='viridis') 
    plt.gca().invert_yaxis() # 将 Y 轴原点翻转到底部
    # ======================================================

    # 绘制白色路径线
    plt.plot(path_x, path_y, 'w-', linewidth=2, label='Optimal Path')
    
    plt.title(f"DTW Path Analysis\n{title}")
    plt.xlabel("Test Sequence Frames")
    plt.ylabel("Template Sequence Frames")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class DTWClassifier:
    def __init__(self):
        self.templates = []
        self.labels = []
        
    def fit(self, X_train, y_train):
        self.templates = X_train
        self.labels = y_train
        
    def predict(self, X_test):
        y_pred = []
        n_test = len(X_test)
        print(f"DTW 预测进度: 0/{n_test}", end='\r')
        
        for idx, test_seq in enumerate(X_test):
            if idx % 5 == 0: print(f"DTW 预测进度: {idx}/{n_test}...", end='\r')
            
            min_dist = float('inf')
            best_label = -1
            
            # 模板匹配：寻找距离最近的训练样本
            for train_seq, label in zip(self.templates, self.labels):
                d, _ = calc_dtw_cost(test_seq, train_seq)
                if d < min_dist:
                    min_dist = d
                    best_label = label
            y_pred.append(best_label)
            
        print(f"DTW 预测进度: {n_test}/{n_test} [完成]")
        return np.array(y_pred)