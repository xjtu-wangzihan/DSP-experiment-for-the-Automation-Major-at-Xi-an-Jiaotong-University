import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
import wave
from exp1_get_wav import read_wav_manual, print_wav_info

# ================= 配置路径 =================
BASE_PATH = r'C:\Users\12427\Desktop\数字信号处理\DSP_test'
RAW_WAV_DIR = os.path.join(BASE_PATH, 'raw_wav')
OUTPUT_DATASET_DIR = os.path.join(BASE_PATH, 'dataset')
PLOT_DIR = os.path.join(BASE_PATH, 'exp1_plots', 'segmentation')

# 确保目录存在
for d in [OUTPUT_DATASET_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ================= 核心算法 =================

def calculate_features(signal, sr, frame_ms=25, step_ms=10):
    """
    计算短时能量和过零率 (手册公式 1 & 2)
    """
    frame_len = int(sr * frame_ms / 1000)
    step_len = int(sr * step_ms / 1000)
    
    signal_len = len(signal)
    if signal_len < frame_len:
        return np.array([]), np.array([]), frame_len, step_len

    num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / step_len))
    
    # 补零
    pad_len = int((num_frames - 1) * step_len + frame_len)
    zeros = np.zeros((pad_len - signal_len,))
    pad_signal = np.concatenate((signal, zeros))
    
    # 向量化分帧
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * step_len, step_len), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32)]
    
    # 1. 短时能量 (Short-time Energy)
    energy = np.sum(frames ** 2, axis=1)
    
    # 2. 过零率 (Zero Crossing Rate)
    signs = np.sign(frames)
    signs[signs == 0] = 1 
    diffs = np.abs(signs[:, 1:] - signs[:, :-1])
    zcr = 0.5 * np.sum(diffs, axis=1)
    
    return energy, zcr, frame_len, step_len

def get_segments_with_params(energy, sr, step_len, coef=0.025, min_gap_ms=150):
    """
    使用指定参数进行一次端点检测尝试
    """
    # 动态计算阈值：底噪 + 系数 * (峰值 - 底噪)
    # 取能量较低的50%作为背景噪声估计
    sorted_energy = np.sort(energy)
    noise_level = np.mean(sorted_energy[:len(energy)//2])
    peak_level = np.max(energy)
    
    # 这里的 coef 是灵敏度关键，越小越灵敏
    threshold = noise_level + coef * (peak_level - noise_level)
    
    is_speech = energy > threshold
    
    segments = []
    start = -1
    
    # 最小语音长度 (太短视为噪声)
    min_duration_frames = int(0.10 * sr / step_len) # 100ms
    # 最小静音间隔 (小于此间隔视为同一个字)
    min_gap_frames = int(min_gap_ms * sr / 1000 / step_len) 
    
    for i in range(len(is_speech)):
        if is_speech[i]:
            if start == -1: start = i
        else:
            if start != -1:
                end = i
                # 逻辑：如果与上一段间隔足够短，则合并
                if len(segments) > 0 and (start - segments[-1][1]) < min_gap_frames:
                    segments[-1] = (segments[-1][0], end) # 合并
                else:
                    segments.append((start, end))
                start = -1
                
    if start != -1: segments.append((start, len(is_speech)))
    
    # 过滤过短片段
    valid_segments = [s for s in segments if (s[1] - s[0]) > min_duration_frames]
    
    return valid_segments, threshold

def adaptive_endpoint_detection(energy, sr, step_len):
    """
    自适应端点检测：如果发现片段过少，自动降低阈值重试。
    """
    # 初始参数尝试
    # coef: 能量阈值系数 (0.04 -> 0.02 -> 0.01)
    # min_gap_ms: 两个字之间的最小间隔 (防止合并)
    
    # 策略列表： (阈值系数, 最小间隙ms)
    # 1. 严格模式 (0.04, 200ms) -> 适合清晰录音
    # 2. 标准模式 (0.02, 150ms) -> 适合一般录音 (原版参数)
    # 3. 灵敏模式 (0.01, 120ms) -> 适合声音小的情况
    # 4. 极度灵敏 (0.005, 100ms) -> 最后的尝试
    strategies = [
        (0.05, 200), 
        (0.025, 150), 
        (0.01, 120),
        (0.005, 100)
    ]
    
    best_segments = []
    best_threshold = 0
    
    for i, (coef, gap) in enumerate(strategies):
        segments, threshold = get_segments_with_params(energy, sr, step_len, coef, gap)
        
        # 这里的逻辑是：我们需要找大约20个数字。
        # 如果找到了 15-25 个，认为比较合理，直接返回
        # 如果少于 10 个，说明阈值太高漏掉了，继续下一轮循环(降低阈值)
        # 如果超过 30 个，说明阈值太低引入了噪声，但这里我们优先保证召回率，取最接近20的一次
        
        count = len(segments)
        # print(f"    [调试] 策略{i+1} (coef={coef}, gap={gap}ms) -> 发现 {count} 个片段")
        
        if 12 <= count <= 25:
            return segments, threshold, step_len # 完美
        
        # 如果是最后一次尝试，或者当前结果比上一次更好（更接近20），则更新最佳结果
        if len(best_segments) == 0 or (abs(count - 20) < abs(len(best_segments) - 20)):
            best_segments = segments
            best_threshold = threshold
            
        # 如果找到的太少，继续循环尝试更低的阈值
        if count < 12:
            continue
        else:
            # 如果找到太多(例如噪音被算进去了)，可能不需要继续降低阈值了，
            # 但为了保险，暂且保留当前结果作为备选
            pass
            
    return best_segments, best_threshold, step_len

def plot_analysis(signal, segments, energy, zcr, threshold, step_len, title, save_path):
    """
    绘制三图合一：原始波形+切割线、短时能量、过零率
    """
    plt.figure(figsize=(10, 10))
    
    # 1. 原始波形
    plt.subplot(3, 1, 1)
    plt.plot(signal, color='#1f77b4')
    plt.title(f'Waveform & Segmentation: {title} (Count: {len(segments)})')
    plt.ylabel('Amplitude')
    for i, (s_frame, e_frame) in enumerate(segments):
        # 将帧索引转换为采样点索引用于绘图
        s = s_frame * step_len
        e = e_frame * step_len
        plt.axvline(s, color='r', linestyle='--', alpha=0.7)
        plt.axvline(e, color='r', linestyle='--', alpha=0.7)
        # 标号
        plt.text((s+e)/2, np.max(signal)*0.9, str(i+1), color='r', ha='center', fontsize=8)

    # 2. 短时能量
    plt.subplot(3, 1, 2)
    t_axis = np.arange(len(energy)) * step_len
    plt.plot(t_axis, energy, color='k')
    plt.axhline(threshold, color='g', linestyle='--', label='Threshold')
    plt.ylabel('Short-time Energy')
    plt.legend()

    # 3. 过零率
    plt.subplot(3, 1, 3)
    plt.plot(t_axis, zcr, color='purple')
    plt.ylabel('Zero Crossing Rate')
    plt.xlabel('Sample Index')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_segments(signal, segments, sr, digit, global_counter, step_len):
    """保存切割片段到 dataset/digit 文件夹"""
    save_dir = os.path.join(OUTPUT_DATASET_DIR, str(digit))
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    saved_count = 0
    padding_frames = 5 
    
    for s_frame, e_frame in segments:
        # 转换回采样点，并稍微外扩一点
        start = max(0, (s_frame - padding_frames) * step_len)
        end = min(len(signal), (e_frame + padding_frames) * step_len)
        
        if end <= start: continue
        
        chunk = signal[start:end]
        
        # 归一化并转 int16 (保存时放大音量，方便听)
        max_val = np.max(np.abs(chunk))
        if max_val > 0:
            chunk = chunk / max_val * 0.9 
        
        chunk_int16 = (chunk * 32767).astype(np.int16)
        
        filename = os.path.join(save_dir, f"{digit}_{global_counter:04d}.wav")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(chunk_int16.tobytes())
        
        global_counter += 1
        saved_count += 1
    return saved_count

# ================= 主流程 =================

if __name__ == "__main__":
    # 清空旧数据 (可选)
    if os.path.exists(OUTPUT_DATASET_DIR): shutil.rmtree(OUTPUT_DATASET_DIR)
    os.makedirs(OUTPUT_DATASET_DIR)

    # 获取所有 _raw.wav 文件
    files = glob.glob(os.path.join(RAW_WAV_DIR, '**', '*_raw.wav'), recursive=True)
    files.sort() # 排序保证处理顺序一致
    
    print(f"找到 {len(files)} 个原始录音文件，开始处理...")
    
    # key: digit (0-9), value: current_count
    digit_counters = {str(i): 0 for i in range(10)}
    
    for file_path in files:
        try:
            # 解析文件名
            filename = os.path.basename(file_path)
            digit = filename.split('_')[0] # "0"
            speaker = os.path.basename(os.path.dirname(file_path)) # "sfm"
            
            # 1. 读取
            signal, sr = read_wav_manual(file_path)
            if signal is None: continue
            
            print_wav_info(file_path, signal, sr)
            
            # 2. 预处理：去直流 + 幅度归一化 (关键修复)
            signal = signal - np.mean(signal)
            max_amp = np.max(np.abs(signal))
            if max_amp > 0:
                signal = signal / max_amp
            
            # 3. 计算特征
            energy, zcr, frame_len, step_len = calculate_features(signal, sr)
            
            if len(energy) == 0:
                print(f"  [警告] 文件 {filename} 太短或为空，跳过")
                continue

            # 4. 自适应端点检测 (关键修复)
            segments, threshold, step_len = adaptive_endpoint_detection(energy, sr, step_len)
            
            count = len(segments)
            warn_str = ""
            if count < 15 or count > 25:
                warn_str = " (!)"
            
            print(f"  -> 说话人: {speaker}, 数字: {digit}, 检测到: {count} 个片段{warn_str}")
            
            # 5. 绘图
            plot_name = f"{speaker}_{digit}_analysis.png"
            plot_analysis(signal, segments, energy, zcr, threshold, step_len, 
                          f"{speaker} - {digit}", 
                          os.path.join(PLOT_DIR, plot_name))
            
            # 6. 保存切片
            save_count = save_segments(signal, segments, sr, digit, digit_counters[digit], step_len)
            digit_counters[digit] += save_count
            
        except Exception as e:
            print(f"  [Error] 处理 {file_path} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n处理完成。")
    print(f"数据集已生成在: {OUTPUT_DATASET_DIR}")
    print(f"可视化图表在: {PLOT_DIR}")
    print("请检查带有 (!) 标记的文件对应的图片，确认切割是否正确。")