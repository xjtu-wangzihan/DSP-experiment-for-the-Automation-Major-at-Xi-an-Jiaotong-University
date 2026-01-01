import numpy as np
import wave
import os

# 引入你之前的读取函数
from get_wav import read_wav_manual

def save_debug_wav(filename, signal, sr):
    """将浮点数信号保存为 int16 WAV"""
    # 确保数据在 -1 到 1 之间
    signal = np.clip(signal, -1.0, 1.0)
    # 转换为 16-bit PCM
    signal_int16 = (signal * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)   # 单声道
        wf.setsampwidth(2)   # 16 bit
        wf.setframerate(sr)
        wf.writeframes(signal_int16.tobytes())
    print(f"已生成文件: {filename}")

if __name__ == "__main__":
    # 1. 设置文件路径
    input_file = '/data2/gyxu/Programs/DSP_test/raw_wav/0_raw.wav'
    
    print(f"正在读取: {input_file}")
    signal, sr = read_wav_manual(input_file)
    
    if signal is None:
        print("读取失败，请检查文件路径。")
        exit()
        
    print(f"原始数据形状: {signal.shape} (样本数, 声道数)")
    
    if signal.ndim < 2 or signal.shape[1] < 2:
        print("警告: 该文件似乎已经是单声道，无法进行混合测试。")
        # 如果本来就是单声道，直接保存一份看看是不是保存过程的问题
        save_debug_wav('debug_original_mono.wav', signal, sr)
    else:
        # 2. 提取不同的声道版本
        
        # [方案 A] 混合声道 (之前代码的逻辑)
        # 公式: (Left + Right) / 2
        mixed_signal = np.mean(signal, axis=1)
        save_debug_wav('debug_mixed.wav', mixed_signal, sr)
        
        # [方案 B] 仅左声道
        left_signal = signal[:, 0]
        save_debug_wav('debug_left.wav', left_signal, sr)
        
        # [方案 C] 仅右声道
        right_signal = signal[:, 1]
        save_debug_wav('debug_right.wav', right_signal, sr)
        
        # [方案 D] 相减 (测试是否反相)
        # 如果 L 和 R 是反相的，相加会抵消，相减反而会增强
        diff_signal = (signal[:, 0] - signal[:, 1]) / 2
        save_debug_wav('debug_diff.wav', diff_signal, sr)

    print("\n=== 分析指南 ===")
    print("1. 请试听 debug_mixed.wav。如果它听起来很怪，但 debug_left.wav 听起来正常：")
    print("   -> 说明左右声道存在相位差，相加导致了抵消。")
    print("   -> 解决方法：在之前的代码中，不要使用 np.mean()，而是直接取 signal[:, 0] (仅用左声道)。")
    print("2. 如果 debug_mixed.wav 听起来正常，但之前的切分文件听起来怪：")
    print("   -> 说明问题出在切分保存时的 writeframes 逻辑上。")
    print("3. 如果 debug_diff.wav 听起来比 mixed 更清晰：")
    print("   -> 这是一个典型的相位反转录音事故（麦克风接反了）。")