import os
import sys
import glob
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score

# ================= 路径配置与依赖导入 =================

# 1. 动态获取路径
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../exp3_programs
parent_dir = os.path.dirname(current_dir)                # .../DSP_test
exp1_dir = os.path.join(parent_dir, 'exp1_programs')     # .../DSP_test/exp1_programs
dataset_dir = os.path.join(parent_dir, 'dataset_exp3')   # .../DSP_test/dataset_exp3
plot_dir = os.path.join(parent_dir, 'exp3_plots')        # .../DSP_test/exp3_plots

# 2. 自动创建绘图目录
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# 3. 导入 exp1_get_wav
if exp1_dir not in sys.path:
    sys.path.append(exp1_dir)

try:
    from exp1_get_wav import read_wav_manual
except ImportError:
    print(f"错误: 无法在 {exp1_dir} 中找到 exp1_get_wav.py")
    sys.exit(1)

# ================= 核心处理模块 =================

def preprocess_audio(file_path, target_sr=16000):
    """
    读取 -> 转单声道 -> 重采样 -> 静音切除 (VAD)
    """
    # 1. 读取
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None, None

    # 2. 转单声道
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    
    # 3. 重采样
    if sr != target_sr:
        # librosa 需要 float 类型
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    # 4. 简单的能量端点检测 (VAD) - 去除静音
    # 目的：只保留有语音的部分，提高声纹纯度
    frame_len = int(sr * 0.03) # 30ms
    hop_len = int(frame_len / 2)
    
    rms = librosa.feature.rms(y=signal, frame_length=frame_len, hop_length=hop_len)[0]
    
    # 动态阈值：取平均能量的 50% 或固定底噪 0.01 的较大者
    threshold = max(0.01, np.mean(rms) * 0.5)
    
    # 使用 librosa.effects.split 切割非静音段
    intervals = librosa.effects.split(signal, top_db=25) # top_db 越大越宽松
    
    clean_signal = np.array([])
    for start, end in intervals:
        clean_signal = np.concatenate((clean_signal, signal[start:end]))
        
    # 如果切得太狠（比如只剩不到0.1秒），说明 VAD 可能误判，退化为使用原信号
    if len(clean_signal) < int(0.1 * sr): 
        return signal, sr
        
    return clean_signal, sr

def extract_features_gmm(file_path):
    """
    提取用于 GMM 的 MFCC 特征
    返回: (N_frames, 40) [包含 Delta 特征]
    """
    signal, sr = preprocess_audio(file_path)
    if signal is None: return None
    
    # 预加重
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # 提取 MFCC (20维)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
    
    # 计算一阶差分 (Delta)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # 堆叠: (40, N_frames)
    combined = np.vstack((mfcc, mfcc_delta))
    
    # 转置为 (N_frames, 40)
    return combined.T

# ================= 模型管理模块 =================

def train_speaker_models(data_dir):
    """
    遍历文件夹，为每个说话人训练一个 GMM
    """
    models = {} 
    
    # 获取子文件夹列表 (即说话人名字)
    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录不存在: {data_dir}")
        return {}

    speakers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"检测到 {len(speakers)} 位说话人: {speakers}")
    print(">>> 开始训练声纹模型 (GMM)...")
    
    for speaker in speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        # 寻找训练文件：文件名包含 'train'
        train_files = glob.glob(os.path.join(speaker_dir, "*train*.wav"))
        
        if not train_files:
            print(f"  [跳过] {speaker}: 未找到训练文件 (需包含 'train' 关键字)")
            continue
            
        print(f"  正在建模: {speaker} (样本数: {len(train_files)})")
        
        # 收集特征
        features = []
        for f in train_files:
            feat = extract_features_gmm(f)
            if feat is not None:
                features.append(feat)
        
        if not features: continue
            
        # 堆叠所有帧
        X_train = np.vstack(features)
        
        # 训练 GMM
        # n_components=16: 用 16 个高斯分布拟合声纹空间
        gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, random_state=42)
        gmm.fit(X_train)
        
        models[speaker] = gmm
        print(f"    -> 模型就绪 (拟合了 {X_train.shape[0]} 帧特征)")
        
    return models

def identify_speaker(file_path, models):
    """
    识别单条语音的说话人
    """
    X_test = extract_features_gmm(file_path)
    if X_test is None: return None, -np.inf
    
    best_score = -np.inf
    best_speaker = None
    
    # 轮询计算对数似然度 (Log-Likelihood)
    for name, model in models.items():
        try:
            score = model.score(X_test)
            if score > best_score:
                best_score = score
                best_speaker = name
        except Exception:
            pass
            
    return best_speaker, best_score

# ================= 主程序 =================

if __name__ == "__main__":
    print(f"当前工作目录: {current_dir}")
    print(f"数据集路径:   {dataset_dir}")
    
    # --- Step 1: 训练 ---
    print("\n" + "="*40)
    print("阶段一: 建立声纹库")
    print("="*40)
    
    speaker_models = train_speaker_models(dataset_dir)
    
    if not speaker_models:
        print("错误: 未训练出任何模型。请检查 dataset_exp3 中是否有包含 'train' 的 wav 文件。")
        sys.exit(1)
        
    # --- Step 2: 测试 ---
    print("\n" + "="*40)
    print("阶段二: 身份识别测试")
    print("="*40)
    
    y_true = []
    y_pred = []
    labels = list(speaker_models.keys())
    
    total_tests = 0
    
    for true_speaker in labels:
        speaker_dir = os.path.join(dataset_dir, true_speaker)
        # 寻找测试文件：文件名包含 'test'
        test_files = glob.glob(os.path.join(speaker_dir, "*test*.wav"))
        
        if not test_files:
            continue
            
        print(f"\n>>> 正在测试说话人: [{true_speaker}]")
        
        for t_file in test_files:
            total_tests += 1
            filename = os.path.basename(t_file)
            
            # 识别
            pred_speaker, score = identify_speaker(t_file, speaker_models)
            
            is_correct = (pred_speaker == true_speaker)
            mark = "✅" if is_correct else f"❌ (误判为 {pred_speaker})"
            
            print(f"  文件: {filename:<20} -> score: {score:.2f} | {mark}")
            
            y_true.append(true_speaker)
            y_pred.append(pred_speaker)
            
    if total_tests == 0:
        print("\n警告: 未找到任何测试文件 (需包含 'test' 关键字)。")
        sys.exit(0)

    # --- Step 3: 结果汇总 ---
    print("\n" + "="*40)
    print("实验3 最终报告")
    print("="*40)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"总体识别准确率: {acc:.2%}")
    
    # 保存混淆矩阵
    if len(y_true) > 0:
        plt.figure(figsize=(8, 6))
        # 确保标签顺序一致
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title('Speaker Identification Confusion Matrix')
        plt.ylabel('True Speaker')
        plt.xlabel('Predicted Speaker')
        plt.tight_layout()
        
        save_path = os.path.join(plot_dir, 'exp3_confusion_matrix.png')
        plt.savefig(save_path)
        print(f"混淆矩阵图片已保存至: {save_path}")

    print("\n[实验原理提示]")
    print("1. 本实验实现了独立于文本的说话人识别 (Text-Independent Speaker Recognition)。")
    print("2. 核心技术: MFCC 特征提取 + GMM-UBM (高斯混合模型) 建模。")
    print("3. GMM 能够拟合声纹特征在多维空间中的概率分布，从而实现对身份的判别。")