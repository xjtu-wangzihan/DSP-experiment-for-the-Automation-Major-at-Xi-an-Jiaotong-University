import numpy as np

def pre_emphasis(signal, alpha=0.97):
    """
    [实验2 原理3] 语音信号的预加重 
    公式: y[n] = x[n] - alpha * x[n-1]
    作用: 提升高频分量，平坦化频谱，便于后续共振峰提取。
    """
    # 在开头补零以保持长度一致，避免维度错误
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])