import os
import glob
import numpy as np
import librosa
import joblib  # 用于保存/加载模型
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score

# 复用之前的读取函数
from get_wav import read_wav_manual

# ==========================================
# 1. 核心处理模块：标准化 + VAD + 特征提取
# ==========================================

def preprocess_audio(file_path, target_sr=16000):
    """
    读取 -> 转单声道 -> 重采样 -> 静音切除 (VAD)
    """
    # 1. 读取 (支持任意格式)
    signal, sr = read_wav_manual(file_path)
    if signal is None: return None, None

    # 2. 转单声道
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    
    # 3. 重采样 (强制统一到 16k)
    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    # 4. 简单的能量端点检测 (VAD) - 去除静音
    # 说话人识别中，静音段不包含身份信息，必须去除
    # 窗口大小 30ms
    frame_len = int(sr * 0.03)
    hop_len = int(frame_len / 2)
    
    # 计算短时能量
    rms = librosa.feature.rms(y=signal, frame_length=frame_len, hop_length=hop_len)[0]
    
    # 设定阈值 (平均能量的 30% 或 一个固定低值)
    threshold = max(0.01, np.mean(rms) * 0.5)
    
    # 找出有声音的帧
    active_frames = rms > threshold
    
    # 简单的帧级掩码转信号级掩码 (这里为了简单，直接利用 librosa 的效果)
    # 使用 librosa.effects.split 更稳健
    intervals = librosa.effects.split(signal, top_db=20) # top_db越小越严格
    
    clean_signal = np.array([])
    for start, end in intervals:
        clean_signal = np.concatenate((clean_signal, signal[start:end]))
        
    if len(clean_signal) < 1000: # 如果切完了没剩啥，就用原信号
        return signal, sr
        
    return clean_signal, sr

def extract_features_gmm(file_path):
    """
    提取用于 GMM 的 MFCC 特征
    返回形状: (N_frames, 20)  <-- 注意：保留时间轴，不求平均
    """
    signal, sr = preprocess_audio(file_path)
    if signal is None: return None
    
    # 预加重
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # 提取 MFCC (提取 20 维以获取更多细节)
    # n_fft=1024, hop_length=256 (16ms 帧移，增加帧数)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
    
    # 增加一阶差分 (Delta) 捕捉动态特性
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # 拼接: (40, N_frames)
    combined = np.vstack((mfcc, mfcc_delta))
    
    # 转置为 (N_frames, 40) 以符合 sklearn 输入要求
    return combined.T

# ==========================================
# 2. 模型管理模块：训练与识别
# ==========================================

def train_speaker_models(dataset_dir):
    """
    遍历文件夹，为每个说话人训练一个 GMM
    """
    models = {} # { "Speaker_1_sfm": gmm_model, ... }
    speakers = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"检测到 {len(speakers)} 位说话人: {speakers}")
    print("开始训练模型 (GMM)...")
    
    for speaker in speakers:
        speaker_dir = os.path.join(dataset_dir, speaker)
        # 寻找训练文件 (包含 train 字样的 wav)
        train_files = glob.glob(os.path.join(speaker_dir, "*train*.wav"))
        
        if not train_files:
            print(f"  [跳过] {speaker}: 未找到训练文件 (*train*.wav)")
            continue
            
        print(f"  正在训练: {speaker} (使用 {len(train_files)} 个文件)...")
        
        # 收集该说话人的所有特征帧
        features = []
        for f in train_files:
            feat = extract_features_gmm(f)
            if feat is not None:
                features.append(feat)
        
        if not features: continue
            
        # 堆叠成一个巨大的矩阵 (Total_Frames, 40)
        X_train = np.vstack(features)
        
        # 定义 GMM 模型
        # n_components=16 是经典配置，表示用 16 个高斯分布来拟合声纹
        gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, random_state=42)
        gmm.fit(X_train)
        
        models[speaker] = gmm
        print(f"    -> 模型训练完毕 (拟合了 {X_train.shape[0]} 帧数据)")
        
    return models

def identify_speaker(file_path, models):
    """
    输入一个测试文件，返回最可能的说话人名字
    """
    # 1. 提取特征
    X_test = extract_features_gmm(file_path)
    if X_test is None: return None, -np.inf
    
    best_score = -np.inf
    best_speaker = None
    
    # 2. 轮询所有模型
    # 原理：计算 Log-Likelihood (对数似然度)
    # 分数越高，表示这段语音越符合该模型的分布
    for name, model in models.items():
        score = model.score(X_test) # 返回平均对数似然
        # print(f"    debug: vs {name} score={score:.2f}") 
        
        if score > best_score:
            best_score = score
            best_speaker = name
            
    return best_speaker, best_score

# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    base_dir = "dataset_exp3"
    
    if not os.path.exists(base_dir):
        print(f"错误: 未找到目录 {base_dir}")
        exit()
        
    # --- Step 1: 训练阶段 ---
    print("========== 阶段一: 说话人声纹建模 ==========")
    speaker_models = train_speaker_models(base_dir)
    
    if not speaker_models:
        print("未训练出任何模型，请检查文件名是否包含 'train'。")
        exit()
        
    # --- Step 2: 测试阶段 ---
    print("\n========== 阶段二: 独立于内容的身份识别测试 ==========")
    
    y_true = []
    y_pred = []
    labels = list(speaker_models.keys())
    
    # 遍历每个文件夹下的测试文件
    for true_speaker in labels:
        speaker_dir = os.path.join(base_dir, true_speaker)
        # 寻找测试文件 (包含 test 字样的 wav)
        test_files = glob.glob(os.path.join(speaker_dir, "*test*.wav"))
        
        for t_file in test_files:
            filename = os.path.basename(t_file)
            print(f"测试文件: [{true_speaker}] / {filename}")
            
            # 识别
            pred_speaker, score = identify_speaker(t_file, speaker_models)
            
            print(f"  -> 识别结果: {pred_speaker} (置信度: {score:.2f})")
            print(f"  -> {'✅ 正确' if pred_speaker == true_speaker else '❌ 错误'}")
            
            y_true.append(true_speaker)
            y_pred.append(pred_speaker)
            
    # --- Step 3: 结果汇总 ---
    print("\n========== 实验结果汇总 ==========")
    acc = accuracy_score(y_true, y_pred)
    print(f"总体识别准确率: {acc:.2%}")
    
    # 绘制简单的混淆矩阵
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Speaker Identification Confusion Matrix')
        plt.ylabel('True Speaker')
        plt.xlabel('Predicted Speaker')
        plt.tight_layout()
        plt.savefig('exp3_speaker_confusion.png')
        print("混淆矩阵已保存为 exp3_speaker_confusion.png")

    print("\n[分析提示]")
    print("1. 本实验采用了 GMM-UBM 的核心思想，利用 MFCC 分布特征进行身份建模。")
    print("2. 'score' 代表对数似然度，数值越大(越接近0)表示越匹配。")
    print("3. 如果出现识别错误，通常是因为录音时长太短(特征覆盖不全)或环境噪音差异过大。")