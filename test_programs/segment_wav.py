import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# 引入你写好的读取函数
from get_wav import read_wav_manual

def to_mono(signal):
    """将立体声转换为单声道"""
    if signal.ndim > 1:
        return np.mean(signal, axis=1)
    return signal

def enframe_signal(signal, frame_len, step_len):
    """分帧"""
    signal_len = len(signal)
    num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / step_len))
    
    # 补零
    pad_len = int((num_frames - 1) * step_len + frame_len)
    zeros = np.zeros((pad_len - signal_len,))
    pad_signal = np.concatenate((signal, zeros))
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * step_len, step_len), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    return pad_signal[indices]

def multi_endpoint_detection(signal, sample_rate, file_label="Unknown"):
    """
    多重端点检测：从长音频中分割出多个语音片段。
    依据：短时能量 
    """
    # 1. 参数设置
    frame_ms = 25
    step_ms = 10
    frame_len = int(sample_rate * frame_ms / 1000)
    step_len = int(sample_rate * step_ms / 1000)
    
    # 2. 计算短时能量
    frames = enframe_signal(signal, frame_len, step_len)
    energy = np.sum(frames ** 2, axis=1)
    
    # 3. 确定动态阈值
    # 假设背景噪声较低，取能量中值或低位分位数作为底噪，最大值的特定比例作为门限
    noise_level = np.mean(energy[energy < np.mean(energy)]) # 估算底噪
    peak_level = np.max(energy)
    # 阈值公式：底噪 + 系数 * (峰值 - 底噪)
    # 这个系数 0.05 需要根据实际录音环境微调
    threshold = noise_level + 0.02 * (peak_level - noise_level)
    
    # 4. 初步筛选 (高于阈值的帧)
    is_speech = energy > threshold
    
    # 5. 寻找连续段 (合并与过滤)
    segments = []
    start = -1
    
    # 辅助参数
    min_duration_ms = 150  # 最短语音长度 (防止切到噪音)
    min_gap_ms = 200       # 两个语音之间的最小间隙 (防止把同一个字切成两半)
    
    min_frames = int(min_duration_ms / step_ms)
    min_gap_frames = int(min_gap_ms / step_ms)
    
    # 遍历帧寻找段落
    for i in range(len(is_speech)):
        if is_speech[i]:
            if start == -1:
                start = i
        else:
            if start != -1:
                end = i
                # 逻辑：如果是新的段，或者与上一段间隔足够远，则添加
                # 否则，合并到上一段
                if len(segments) > 0 and (start - segments[-1][1]) < min_gap_frames:
                    # 合并：更新上一段的结束点
                    segments[-1] = (segments[-1][0], end)
                else:
                    # 添加新段
                    segments.append((start, end))
                start = -1
    
    # 处理最后一段
    if start != -1:
        segments.append((start, len(is_speech)))

    # 6. 过滤过短的段
    valid_segments = [s for s in segments if (s[1] - s[0]) > min_frames]
    
    # 7. 转换为采样点索引
    time_segments = []
    # 稍微向外扩展一点(前后各加5帧)，保证语音完整
    padding = 5
    for s in valid_segments:
        s_idx = max(0, (s[0] - padding)) * step_len
        e_idx = min(len(signal), (s[1] + padding) * step_len + frame_len)
        time_segments.append((s_idx, e_idx))
        
    print(f"文件 {file_label}: 检测到 {len(time_segments)} 个片段")
    
    return time_segments, energy, threshold, step_len

def plot_segmentation(signal, segments, energy, threshold, step_len, file_name):
    """
    可视化端点检测结果，复刻手册图1效果 
    """
    plt.figure(figsize=(12, 8))
    
    # 子图1：原始波形与切割线
    plt.subplot(2, 1, 1)
    plt.plot(signal, color='#0050ef', alpha=0.8) # 蓝色波形
    plt.title(f'Waveform Segmentation: {file_name}', fontsize=12)
    plt.ylabel('Amplitude')
    
    # 绘制红色分割线
    for idx, (start, end) in enumerate(segments):
        plt.axvline(start, color='r', linestyle='-', alpha=0.6, linewidth=1.5)
        plt.axvline(end, color='r', linestyle='-', alpha=0.6, linewidth=1.5)
        # 可选：在每个段上方标注序号
        mid = (start + end) // 2
        plt.text(mid, np.max(signal)*0.9, str(idx+1), color='r', ha='center')

    # 子图2：短时能量与阈值
    plt.subplot(2, 1, 2)
    # 能量图横坐标需要对齐到采样点
    time_axis = np.arange(len(energy)) * step_len
    plt.plot(time_axis, energy, color='black', linewidth=1)
    plt.axhline(threshold, color='g', linestyle='--', label='Threshold')
    plt.title('Short-Time Energy', fontsize=12)
    plt.xlabel('Sample Index')
    plt.ylabel('Energy')
    plt.legend()
    
    plt.tight_layout()
    # 保存图片以便查看
    output_img = f'segmentation_result_{file_name}.png'
    plt.savefig(output_img)
    print(f"可视化图表已保存为: {output_img}")
    # plt.show() # 如果在服务器端运行，请注释此行

def save_sliced_audio(signal, segments, sr, digit_label, output_base_dir="dataset"):
    """保存切割后的音频片段"""
    import struct
    
    save_dir = os.path.join(output_base_dir, str(digit_label))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) # 清空旧数据
    os.makedirs(save_dir)
    
    for i, (start, end) in enumerate(segments):
        # 获取片段
        chunk = signal[start:end]
        
        # 归一化并转回 PCM 16-bit
        # 假设读取时归一化到了[-1, 1]，保存时需乘 32767
        chunk_int16 = (chunk * 32767).astype(np.int16)
        
        filename = os.path.join(save_dir, f"{digit_label}_{i+1:02d}.wav")
        
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1) # 单声道
                wf.setsampwidth(2) # 16 bit = 2 bytes
                wf.setframerate(sr)
                wf.writeframes(chunk_int16.tobytes())
        except NameError:
            import wave
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(chunk_int16.tobytes())
                
    print(f"已保存 {len(segments)} 个样本到 {save_dir}")

# ===========================
# 主执行流程
# ===========================

if __name__ == "__main__":
    base_path = '/data2/gyxu/Programs/DSP_test/raw_wav'
    summary = [] # 用于存储处理结果统计
    
    print("================ 开始批量处理 ================")
    
    # 循环 0 到 9
    for i in range(10):
        digit_label = str(i)
        filename = f"{digit_label}_raw.wav"
        full_path = os.path.join(base_path, filename)
        
        print(f"\n正在处理: {filename} ...")
        
        if not os.path.exists(full_path):
            print(f"  [错误] 文件不存在: {full_path}")
            summary.append((digit_label, "文件缺失"))
            continue
            
        try:
            # 1. 读取
            signal, sr = read_wav_manual(full_path)
            if signal is None:
                summary.append((digit_label, "读取失败"))
                continue
                
            # 2. 预处理
            signal = to_mono(signal)
            signal = signal - np.mean(signal)
            
            # 3. 检测
            segments, energy, threshold, step_len = multi_endpoint_detection(
                signal, sr, file_label=digit_label
            )
            
            # 4. 绘图
            plot_segmentation(signal, segments, energy, threshold, step_len, digit_label)
            
            # 5. 保存
            save_sliced_audio(signal, segments, sr, digit_label)
            
            # 记录统计
            summary.append((digit_label, f"{len(segments)} 个片段"))
            
        except Exception as e:
            print(f"  [异常] 处理出错: {e}")
            summary.append((digit_label, f"出错: {str(e)}"))

    print("\n================ 处理结果汇总 ================")
    print(f"{'数字':<6} | {'状态/数量':<15}")
    print("-" * 25)
    for label, status in summary:
        # 如果数量不是20，标记一下方便查看
        mark = "" 
        if "20" not in status and "个片段" in status:
            mark = " (!)" 
        print(f"{label:<6} | {status:<15}{mark}")
    print("==============================================")
    print("提示：带有 (!) 标记的数字可能需要微调阈值参数。")