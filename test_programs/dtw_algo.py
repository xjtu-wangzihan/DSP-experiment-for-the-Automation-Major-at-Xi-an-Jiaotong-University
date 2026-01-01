import numpy as np

def calc_dtw_distance(seq1, seq2):
    """
    计算两个不等长序列之间的 DTW 距离 
    seq1: (N, D)
    seq2: (M, D)
    """
    n, d1 = seq1.shape
    m, d2 = seq2.shape
    
    # 距离矩阵 (欧氏距离)
    # dist_matrix[i, j] = ||seq1[i] - seq2[j]||
    # 使用广播机制加速计算
    diff = seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # 累积代价矩阵 (Accumulated Cost Matrix)
    cost = np.zeros((n, m))
    cost[0, 0] = dist_matrix[0, 0]
    
    # 初始化第一行和第一列
    for i in range(1, n):
        cost[i, 0] = cost[i-1, 0] + dist_matrix[i, 0]
    for j in range(1, m):
        cost[0, j] = cost[0, j-1] + dist_matrix[0, j]
        
    # 动态规划填表 
    # D(i, j) = dist(i, j) + min(D(i-1, j), D(i, j-1), D(i-1, j-1))
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = dist_matrix[i, j] + min(cost[i-1, j], 
                                                 cost[i, j-1], 
                                                 cost[i-1, j-1])
            
    # 返回最终归一化距离
    return cost[n-1, m-1] / (n + m)

class DTWClassifier:
    """
    基于 DTW 距离的最近邻分类器
    """
    def __init__(self):
        self.X_train = []
        self.y_train = []
        
    def fit(self, X_seq_list, y):
        """X_seq_list 是一个列表，包含不同长度的数组"""
        self.X_train = X_seq_list
        self.y_train = y
        
    def predict(self, X_test_seq_list):
        y_pred = []
        total = len(X_test_seq_list)
        print(f"DTW 正在计算 (共 {total} 个样本，速度较慢请耐心等待)...")
        
        for idx, test_seq in enumerate(X_test_seq_list):
            if idx % 10 == 0: print(f"  已处理 {idx}/{total}...")
            
            min_dist = float('inf')
            best_label = -1
            
            # 遍历所有训练样本 (模板匹配)
            for train_seq, train_label in zip(self.X_train, self.y_train):
                dist = calc_dtw_distance(test_seq, train_seq)
                if dist < min_dist:
                    min_dist = dist
                    best_label = train_label
            
            y_pred.append(best_label)
        return np.array(y_pred)