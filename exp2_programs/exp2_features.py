import numpy as np
import librosa
from exp2_utils import pre_emphasis, get_paths

def standardize_audio(signal, current_sr, target_sr=16000):
    """
    音频标准化：统一转单声道并重采样，确保不同设备录音的一致性。
    """
    # 1. 统一转单声道 (Handle Stereo)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    
    # 2. 统一重采样 (Resample)
    if current_sr != target_sr:
        # librosa.resample 需要 float 输入，read_wav_manual 已经输出 float
        signal = librosa.resample(signal, orig_sr=current_sr, target_sr=target_sr)
        
    return signal, target_sr

def extract_mfcc(file_path, n_mfcc=13, return_raw=False):
    """
    [实验2 原理2] 计算 Mel 频率倒谱系数 (MFCC)
    """
    paths = get_paths()
    read_wav = paths['raw_reader']
    
    # 1. 读取
    signal, sr = read_wav(file_path)
    if signal is None: return None
    
    # 2. 标准化 (重要：MFCC 参数依赖于采样率)
    target_sr = 16000
    signal, sr = standardize_audio(signal, sr, target_sr=target_sr)
    
    # 3. 预加重
    signal = pre_emphasis(signal)
    
    # 4. 计算 MFCC
    # n_fft=512, hop_length=256 对应约 32ms 窗长和 16ms 帧移 (at 16kHz)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    
    if return_raw:
        return mfcc
    
    # 5. 倒谱均值归一化 (CMN) - 消除信道影响
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True))
    
    return mfcc.T # 转置为 (Frames, n_mfcc) 用于 DTW

def get_feature_stats(mfcc_seq):
    """
    提取统计特征供传统分类器 (SVM/LDA) 使用
    输入: (Frames, n_mfcc)
    输出: (2 * n_mfcc, ) -> [Mean, Std]
    """
    mean = np.mean(mfcc_seq, axis=0)
    std = np.std(mfcc_seq, axis=0)
    return np.concatenate([mean, std])