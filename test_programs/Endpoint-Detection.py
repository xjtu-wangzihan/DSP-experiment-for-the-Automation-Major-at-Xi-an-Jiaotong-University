import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob

# 引入你写好的读取函数
from get_wav import read_wav_manual

# ==========================================
# 1. 基础工具函数
# ==========================================

def to_mono(signal):
    """
    将立体声信号转换为单声道。
    语音识别通常只需要单声道数据。
    """
    if signal.ndim > 1:
        # 对多声道取平均值
        return np.mean(signal, axis=1)
    return signal

def pre_emphasis(signal, alpha=0.97):
    """
    预加重 [cite: 151]
    虽然主要是频域实验(实验2)强调预加重，但时域分析中提升高频分量
    有助于提高过零率特征的区分度。
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def enframe(signal, frame_len, step_len):
    """
    分帧函数。
    将连续信号切割成重叠的帧。
    """
    signal_len = len(signal)
    if signal_len <= frame_len:
        return np.array([signal])
    
    num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / step_len))
    pad_len = int((num_frames - 1) * step_len + frame_len)
    zeros = np.zeros((pad_len - signal_len,))
    pad_signal = np.concatenate((signal, zeros))
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * step_len, step_len), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    return frames

# ==========================================
# 2. 核心算法实现 (对应实验手册步骤 3 & 4)
# ==========================================

def calculate_short_time_energy(frames):
    """
    计算短时能量 En [cite: 114, 115]
    En = sum(x(m)^2)
    """
    # axis=1 表示对每一帧内部的所有采样点求和
    return np.sum(frames ** 2, axis=1)

def calculate_zcr(frames):
    """
    计算过零率 Zn [cite: 117, 118]
    Zn = 0.5 * sum(|sgn[x(m)] - sgn[x(m-1)]|)
    """
    # 既然已经分帧，我们需要计算每一帧内部的过零次数
    # 为了计算差分，我们在每帧前拼接上一帧的最后一个点（近似处理：这里直接帧内计算）
    # 更加严谨的做法是在分帧前做符号运算，但帧内计算通常足够
    
    # 获取符号函数
    signs = np.sign(frames)
    # 计算相邻点符号差的绝对值
    diffs = np.abs(signs[:, 1:] - signs[:, :-1])
    # 求和并除以2
    zcr = 0.5 * np.sum(diffs, axis=1)
    return zcr

def endpoint_detection(signal, sample_rate, plot=False):
    """
    实验步骤3: 语音信号的预处理(端点检测) 
    基于双门限法（能量 + 过零率）的简化版本：主要基于能量。
    """
    # 参数设置
    frame_size = int(0.025 * sample_rate) # 25ms
    frame_step = int(0.010 * sample_rate) # 10ms
    
    # 分帧
    frames = enframe(signal, frame_size, frame_step)
    
    # 计算特征
    energy = calculate_short_time_energy(frames)
    
    # 简单的能量门限检测
    # 阈值设为：静音能量平均值 + 0.1 * (最大能量 - 静音能量平均值)
    # 假设前5帧是静音
    silence_energy = np.mean(energy[:5])
    threshold = silence_energy + 0.05 * (np.max(energy) - silence_energy)
    
    # 寻找大于阈值的起始和结束帧
    is_speech = energy > threshold
    # 简单的平滑处理 (膨胀)
    # 实际工程中需要更复杂的逻辑(如最短语音长度判断)，这里演示原理
    speech_frames = np.where(is_speech)[0]
    
    if len(speech_frames) > 0:
        start_frame = speech_frames[0]
        end_frame = speech_frames[-1]
        
        # 转换回采样点索引
        start_index = start_frame * frame_step
        end_index = end_frame * frame_step + frame_size
    else:
        start_index = 0
        end_index = len(signal)

    cropped_signal = signal[start_index:end_index]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title('Original Signal & Endpoint Detection')
        plt.plot(signal)
        plt.axvline(start_index, color='r', linestyle='--', label='Start')
        plt.axvline(end_index, color='r', linestyle='--', label='End')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.title('Short-Time Energy')
        plt.plot(energy)
        plt.axhline(threshold, color='g', linestyle='--', label='Threshold')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return cropped_signal

# ==========================================
# 3. 特征提取与分类 (对应实验手册步骤 5 & 6)
# ==========================================

def extract_features_for_classification(file_path):
    """
    为了使用SVM/KNN等分类器，我们需要将变长的语音信号转换为定长的特征向量。
    方法：计算整个语音段的能量和过零率的统计特征（均值、方差、最大值等）。
    """
    # 1. 读取
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None
    
    # 2. 转单声道
    signal = to_mono(signal)
    
    # 3. 预处理：去直流 
    signal = signal - np.mean(signal)
    
    # 4. 端点检测 (切除静音)
    signal = endpoint_detection(signal, sr, plot=False)
    
    if len(signal) < 100: # 信号太短忽略
        return None
        
    # 5. 分帧计算特征
    frame_size = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    frames = enframe(signal, frame_size, frame_step)
    
    energy = calculate_short_time_energy(frames)
    zcr = calculate_zcr(frames)
    
    # 6. 构造定长特征向量 (Statistical Features)
    # 包含：能量均值, 能量方差, 过零率均值, 过零率方差, 信号总时长
    feat_vec = [
        np.mean(energy),
        np.std(energy),
        np.max(energy),
        np.mean(zcr),
        np.std(zcr),
        np.max(zcr),
        len(signal) / sr # 时长
    ]
    return np.array(feat_vec)

# ==========================================
# 主程序：执行单个文件测试
# ==========================================

if __name__ == "__main__":
    # 测试文件路径
    test_file = '/data2/gyxu/Programs/DSP_test/raw_wav/0_raw.wav'
    
    print(f"正在处理文件: {test_file}")
    audio_data, sample_rate = read_wav_manual(test_file)
    
    if audio_data is not None:
        # 转单声道
        mono_signal = to_mono(audio_data)
        print(f"数据形状转换: {audio_data.shape} -> {mono_signal.shape}")
        
        # 执行步骤 3 & 4 的可视化
        print("执行端点检测与时域分析...")
        # 注意：这里会弹窗显示波形图和能量图，展示端点检测效果
        clean_signal = endpoint_detection(mono_signal, sample_rate, plot=True)
        
        print(f"原始时长: {len(mono_signal)/sample_rate:.2f}s")
        print(f"剪切后时长: {len(clean_signal)/sample_rate:.2f}s")
        
        # 演示步骤 5: 分类特征提取
        features = extract_features_for_classification(test_file)
        print("提取的分类特征向量 (Energy_Mean, Energy_Std, ..., ZCR_Mean, ...):")
        print(features)
        
        print("\n---------------------------")
        print("下一步操作建议：")
        print("要完成实验手册的【分类器实现】(步骤5),你需要对0-9每个数字")
        print("采集10个以上的样本。请将它们整理在文件夹中，例如：")
        print("dataset/0/0_01.wav, dataset/0/0_02.wav ... dataset/1/1_01.wav ...")
        print("然后我可以为你提供遍历文件夹进行批量训练的代码。")