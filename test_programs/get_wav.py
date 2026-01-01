import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取WAV文件头并手动解析数据
def read_wav_manual(filename):
    """
    手动解析WAV文件头并读取数据。
    修正版：支持多声道数据的自动重塑 (Reshape)。
    """
    try:
        with open(filename, 'rb') as f:
            # --- 1. 解析文件头 ---
            riff_chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            format_type = f.read(4)

            chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            
            # 寻找 fmt 块
            while chunk_id != b'fmt ':
                f.seek(chunk_size, 1)
                chunk_id = f.read(4)
                if not chunk_id: raise ValueError('未找到 fmt 块')
                chunk_size = struct.unpack('<I', f.read(4))[0]
            
            fmt_chunk_start = f.tell()
            audio_format = struct.unpack('<H', f.read(2))[0]
            num_channels = struct.unpack('<H', f.read(2))[0]  # <--- 关键：获取声道数
            sample_rate = struct.unpack('<I', f.read(4))[0]
            byte_rate = struct.unpack('<I', f.read(4))[0]
            block_align = struct.unpack('<H', f.read(2))[0]
            bits_per_sample = struct.unpack('<H', f.read(2))[0]
            
            # 跳过 fmt 块剩余部分
            bytes_read = f.tell() - fmt_chunk_start
            if chunk_size > bytes_read:
                f.seek(chunk_size - bytes_read, 1)

            # 寻找 data 块
            chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            while chunk_id != b'data':
                f.seek(chunk_size, 1)
                chunk_id = f.read(4)
                if not chunk_id: raise ValueError('未找到 data 块')
                chunk_size = struct.unpack('<I', f.read(4))[0]
            
            data_bytes = f.read(chunk_size)
            
            # --- 2. 将二进制转换为数值 ---
            if bits_per_sample == 16:
                # 16-bit int -> float [-1, 1]
                num_samples = len(data_bytes) // 2
                data = np.frombuffer(data_bytes, dtype=np.int16)
                audio_data = data / 32768.0
            elif bits_per_sample == 32 and audio_format == 3:
                # 32-bit float
                audio_data = np.frombuffer(data_bytes, dtype=np.float32)
            elif bits_per_sample == 8:
                # 8-bit uint -> float [-1, 1]
                data = np.frombuffer(data_bytes, dtype=np.uint8)
                audio_data = (data.astype(np.float32) - 128) / 128.0
            else:
                raise ValueError(f"暂不支持的位深: {bits_per_sample}-bit")

            # --- 3. 关键修正：根据声道数重塑数组 ---
            if num_channels > 1:
                # 如果总点数不能被声道整除，说明数据有损，截断多余的
                total_samples = len(audio_data)
                valid_samples = (total_samples // num_channels) * num_channels
                audio_data = audio_data[:valid_samples]
                
                # Reshape: (N_samples, N_channels)
                audio_data = audio_data.reshape(-1, num_channels)
                
            return audio_data, sample_rate
            
    except Exception as e:
        print(f"读取文件 {filename} 失败: {e}")
        return None, None

# 检查WAV文件头信息
def inspect_wav(filename):
    try:
        with open(filename, 'rb') as f:
            riff = f.read(4)
            size = struct.unpack('<I', f.read(4))[0]
            wave = f.read(4)
            cid = f.read(4)
            csz = struct.unpack('<I', f.read(4))[0]
            while cid != b'fmt ':
                f.seek(csz, 1)
                cid = f.read(4)
                if not cid:
                    print('未找到 fmt 块')
                    return
                csz = struct.unpack('<I', f.read(4))[0]
            fmt_size = csz
            audio_format = struct.unpack('<H', f.read(2))[0]
            num_channels = struct.unpack('<H', f.read(2))[0]
            sample_rate = struct.unpack('<I', f.read(4))[0]
            byte_rate = struct.unpack('<I', f.read(4))[0]
            block_align = struct.unpack('<H', f.read(2))[0]
            bits_per_sample = struct.unpack('<H', f.read(2))[0]
            valid_bits_per_sample = None
            if fmt_size > 16:
                ext_bytes = fmt_size - 16
                if ext_bytes >= 2:
                    cb_size = struct.unpack('<H', f.read(2))[0]
                    ext_bytes -= 2
                    if cb_size >= 2 and ext_bytes >= 2:
                        valid_bits_per_sample = struct.unpack('<H', f.read(2))[0]
                        ext_bytes -= 2
                    if cb_size >= 6 and ext_bytes >= 4:
                        _ = f.read(4)
                        ext_bytes -= 4
                    if cb_size >= 22 and ext_bytes >= 16:
                        _ = f.read(16)
                        ext_bytes -= 16
                if ext_bytes > 0:
                    f.seek(ext_bytes, 1)
            effective_bits = valid_bits_per_sample if valid_bits_per_sample else bits_per_sample
            fmt_name = 'PCM' if audio_format == 1 else ('IEEE_FLOAT' if audio_format == 3 else ('WAVE_EXTENSIBLE' if audio_format == 65534 else f'其他({audio_format})'))
            cid = f.read(4)
            csz = struct.unpack('<I', f.read(4))[0]
            while cid != b'data':
                f.seek(csz, 1)
                cid = f.read(4)
                if not cid:
                    print('未找到 data 块')
                    return
                csz = struct.unpack('<I', f.read(4))[0]
            data_bytes = csz
            frames = data_bytes // block_align if block_align else 0
            print(f'文件: {filename}')
            print(f'容器: {"RIFF" if riff == b"RIFF" else str(riff)} {"WAVE" if wave == b"WAVE" else str(wave)}')
            print(f'格式: {fmt_name}')
            print(f'声道: {num_channels}')
            print(f'采样率: {sample_rate}')
            print(f'位深: {effective_bits}')
            print(f'数据字节数: {data_bytes}')
            print(f'帧数: {frames}')
    except Exception as e:
        print(f'读取文件 {filename} 失败: {e}')

# 测试读取一个文件
signal, sr = read_wav_manual('/data2/gyxu/Programs/DSP_test/raw_wav/0_raw.wav')
if signal is not None and sr is not None:
    print(f"采样率: {sr}, 数据长度: {len(signal)}")
else:
    print('读取失败')
# 检查WAV文件头信息
# inspect_wav('/data2/gyxu/Programs/DSP_test/raw_wav/0_raw.wav')