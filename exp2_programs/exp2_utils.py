import sys
import os
import numpy as np

# ================= 关键路径修正 =================
# 获取当前脚本所在目录 (.../exp2_programs)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父级目录 (.../DSP_test)
parent_dir = os.path.dirname(current_dir)
# 定位实验1脚本目录 (.../DSP_test/exp1_programs)
exp1_dir = os.path.join(parent_dir, 'exp1_programs')

# 将实验1目录加入系统路径，以便导入模块
if exp1_dir not in sys.path:
    sys.path.append(exp1_dir)

try:
    from exp1_get_wav import read_wav_manual
except ImportError:
    print(f"错误: 无法在 {exp1_dir} 中找到 exp1_get_wav.py")
    print("请检查路径结构是否为: /data2/gyxu/Programs/DSP_test/exp1_programs/exp1_get_wav.py")
    sys.exit(1)

def pre_emphasis(signal, alpha=0.97):
    """
    [实验2 原理3] 语音信号的预加重
    公式: y[n] = x[n] - alpha * x[n-1]
    """
    # 保持长度一致
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def get_paths():
    """统一管理路径"""
    paths = {
        'dataset': os.path.join(parent_dir, 'dataset'),       # .../DSP_test/dataset
        'plots': os.path.join(parent_dir, 'exp2_plots'),      # .../DSP_test/exp2_plots
        'raw_reader': read_wav_manual
    }
    
    # 自动创建绘图目录
    if not os.path.exists(paths['plots']):
        os.makedirs(paths['plots'])
        
    return paths