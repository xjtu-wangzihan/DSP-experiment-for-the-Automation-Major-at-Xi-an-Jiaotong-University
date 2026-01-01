import numpy as np
import librosa
from exp2_preproc import pre_emphasis
from get_wav import read_wav_manual

def standardize_audio(signal, current_sr, target_sr=16000):
    """
    [新增] 音频标准化适配器
    1. 统一转单声道
    2. 统一重采样
    """
    # 1. 统一转单声道 (Handle Mono/Stereo)
    if signal.ndim > 1:
        # 如果是双声道 (N, 2)，取平均或取左声道
        signal = np.mean(signal, axis=1)
    # 如果已经是单声道 (N,)，则不动
    
    # 2. 统一重采样 (Resample)
    if current_sr != target_sr:
        # 使用 librosa 进行重采样
        # 注意: librosa 要求输入是 float 类型，read_wav_manual 已经做到了
        signal = librosa.resample(signal, orig_sr=current_sr, target_sr=target_sr)
        
    return signal, target_sr

def extract_mfcc_stats(file_path, n_mfcc=13):
    """[轨道 A] 提取 MFCC 统计特征"""
    # 1. 读取 (可能格式各异)
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None
    
    # 2. === 关键步骤：标准化 ===
    # 强制将所有人的录音统一为 16000Hz 单声道
    signal, sr = standardize_audio(signal, sr, target_sr=16000)
    
    # 3. 后续处理保持不变
    signal = signal - np.mean(signal)
    signal = pre_emphasis(signal)
    
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
    
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)
    
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std])

def extract_mfcc_sequence(file_path, n_mfcc=13):
    """[轨道 B] 提取 MFCC 序列 (DTW用)"""
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None
    
    # === 关键步骤：标准化 ===
    signal, sr = standardize_audio(signal, sr, target_sr=16000)
    
    signal = signal - np.mean(signal)
    signal = pre_emphasis(signal)
    
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
    return mfcc.T