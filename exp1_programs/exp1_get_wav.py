import struct
import numpy as np
import os

def read_wav_manual(filename):
    """
    手动解析WAV文件头并读取数据，返回归一化的单声道信号和采样率。
    符合实验手册要求的"对其中语音数据字段的读取功能"。
    """
    try:
        with open(filename, 'rb') as f:
            # --- 1. 解析文件头 (RIFF chunk) ---
            riff_chunk_id = f.read(4)
            f.read(4) # chunk_size
            format_type = f.read(4)
            if riff_chunk_id != b'RIFF' or format_type != b'WAVE':
                raise ValueError("这不是一个标准的WAVE文件")

            # --- 2. 寻找 fmt 和 data 块 ---
            # 循环寻找子块，增强鲁棒性
            fmt_data = None
            data_bytes = None
            
            while True:
                chunk_id = f.read(4)
                if not chunk_id: break
                chunk_size = struct.unpack('<I', f.read(4))[0]
                
                if chunk_id == b'fmt ':
                    fmt_data = f.read(chunk_size)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    break # 找到数据后通常就不往后读了
                else:
                    f.seek(chunk_size, 1) # 跳过未知块

            if not fmt_data or not data_bytes:
                raise ValueError("文件缺失 fmt 或 data 块")

            # 解析 fmt
            audio_format = struct.unpack('<H', fmt_data[0:2])[0]
            num_channels = struct.unpack('<H', fmt_data[2:4])[0]
            sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
            bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

            # --- 3. 解析数据并归一化 ---
            if bits_per_sample == 16:
                data = np.frombuffer(data_bytes, dtype=np.int16)
                audio_data = data / 32768.0
            elif bits_per_sample == 32: # 兼容浮点格式
                audio_data = np.frombuffer(data_bytes, dtype=np.float32)
            elif bits_per_sample == 8:
                data = np.frombuffer(data_bytes, dtype=np.uint8)
                audio_data = (data.astype(np.float32) - 128) / 128.0
            else:
                raise ValueError(f"暂不支持的位深: {bits_per_sample}")

            # --- 4. 多声道转单声道 ---
            if num_channels > 1:
                # 确保数据长度完整
                num_samples = len(audio_data) // num_channels
                audio_data = audio_data[:num_samples*num_channels]
                audio_data = audio_data.reshape(-1, num_channels)
                # 取平均值转为单声道
                audio_data = np.mean(audio_data, axis=1)

            return audio_data, sample_rate

    except Exception as e:
        print(f"[读取错误] {filename}: {e}")
        return None, None

def print_wav_info(filename, signal, sr):
    """打印文件的基本格式信息，符合实验手册要求"""
    if signal is not None:
        duration = len(signal) / sr
        print(f"文件: {os.path.basename(filename)} | 采样率: {sr}Hz | 时长: {duration:.2f}s | 样本数: {len(signal)}")