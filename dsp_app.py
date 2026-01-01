import streamlit as st
import os
import subprocess
import glob
import time

# ================= 配置：绝对路径 =================
BASE_PATH = '/data2/gyxu/Programs/DSP_test'

# 路径映射字典
PATHS = {
    'exp1': {
        'name': '实验1：时域分析',
        'program_dir': os.path.join(BASE_PATH, 'exp1_programs'),
        'scripts': ['exp1_segment_wav.py', 'exp1_compare_classifiers.py'],
        'data_dir': os.path.join(BASE_PATH, 'raw_wav'),
        'plot_dir': os.path.join(BASE_PATH, 'exp1_plots'),
        'description': "基于短时能量和过零率的端点检测，以及多种分类器的时域特征分类。"
    },
    'exp2': {
        'name': '实验2：频域分析',
        'program_dir': os.path.join(BASE_PATH, 'exp2_programs'),
        'scripts': ['exp2_main.py'],
        'data_dir': os.path.join(BASE_PATH, 'dataset'),
        'plot_dir': os.path.join(BASE_PATH, 'exp2_plots'),
        'description': "MFCC 特征提取，Mel 滤波器组可视化，以及基于 DTW 的序列匹配。"
    },
    'exp3': {
        'name': '实验3：说话人识别',
        'program_dir': os.path.join(BASE_PATH, 'exp3_programs'),
        'scripts': ['exp3_speaker_id_enhanced.py'], 
        'data_dir': os.path.join(BASE_PATH, 'dataset_exp3'),
        'plot_dir': os.path.join(BASE_PATH, 'exp3_plots'),
        'description': "基于 GMM-UBM 的独立于文本的说话人识别系统。"
    }
}

# ================= 辅助函数 =================

def run_script_realtime(script_path, cwd):
    """在网页上实时执行脚本并显示输出"""
    st.info(f"正在启动脚本: {os.path.basename(script_path)} ...")
    
    # 创建一个占位符用于实时更新日志
    log_placeholder = st.empty()
    logs = []
    
    try:
        # 使用 subprocess.Popen 实时捕获输出
        process = subprocess.Popen(
            ['python', script_path],
            cwd=cwd,  # 关键：设置工作目录，确保相对导入正确
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误也重定向到输出
            text=True,
            bufsize=1
        )
        
        # 逐行读取输出
        for line in process.stdout:
            logs.append(line)
            # 实时更新代码块，保留最后 20 行以防刷屏太快，或者显示全部
            log_placeholder.code("".join(logs), language='bash')
        
        process.wait()
        
        if process.returncode == 0:
            st.success("脚本执行完毕！")
        else:
            st.error("脚本执行出错，请检查上方日志。")
            
    except Exception as e:
        st.error(f"运行失败: {e}")

def show_file_browser(data_dir, key_prefix):
    """简单的文件浏览器，展示原始数据"""
    st.markdown("### 📂 原始数据预览")
    st.text(f"数据源路径: {data_dir}")
    
    # 递归查找 wav 文件
    files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    files = sorted(files)[:50] # 限制显示前50个，防止卡顿
    
    if not files:
        st.warning("未找到 .wav 文件")
        return

    selected_file = st.selectbox("选择一个音频文件进行试听:", files, format_func=lambda x: os.path.relpath(x, data_dir), key=f"{key_prefix}_file")
    
    if selected_file:
        st.audio(selected_file)
        file_stats = os.stat(selected_file)
        st.caption(f"文件大小: {file_stats.st_size / 1024:.2f} KB | 路径: {selected_file}")

def show_gallery(plot_dir):
    """展示结果图片画廊"""
    st.markdown("### 📊 实验结果可视化")
    
    if not os.path.exists(plot_dir):
        st.warning(f"图片目录不存在: {plot_dir}")
        return

    # 查找 png 图片
    images = glob.glob(os.path.join(plot_dir, '**', '*.png'), recursive=True)
    
    if not images:
        st.info("暂无生成的图片，请先运行脚本。")
        return
    
    # 分类展示
    confusion_matrices = [img for img in images if "confusion" in os.path.basename(img).lower() or "cm_" in os.path.basename(img)]
    analysis_plots = [img for img in images if img not in confusion_matrices]
    
    # 1. 混淆矩阵 (通常最重要)
    if confusion_matrices:
        st.subheader("1. 分类结果 (混淆矩阵)")
        cols = st.columns(min(3, len(confusion_matrices)))
        for idx, img_path in enumerate(confusion_matrices):
            with cols[idx % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

    # 2. 其他分析图
    if analysis_plots:
        st.subheader("2. 过程分析图表")
        # 增加一个过滤器
        filter_text = st.text_input("筛选图片文件名 (例如: 'sfm', 'pca')", "")
        
        filtered_plots = [p for p in analysis_plots if filter_text.lower() in os.path.basename(p).lower()]
        
        # 分页展示防止卡顿
        batch_size = 9 # 每页显示9张
        total_pages = (len(filtered_plots) - 1) // batch_size + 1
        page = st.number_input("页码", min_value=1, max_value=max(1, total_pages), value=1)
        
        start_idx = (page - 1) * batch_size
        end_idx = start_idx + batch_size
        current_batch = filtered_plots[start_idx:end_idx]
        
        cols = st.columns(3)
        for idx, img_path in enumerate(current_batch):
            with cols[idx % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

# ================= 页面布局 =================

st.set_page_config(page_title="DSP 实验展示平台", layout="wide", page_icon="📈")

st.title("数字信号处理实验展示系统")
st.markdown("**学生**: 孙凤鸣 | **服务器**: Linux Server2")
st.markdown("---")

# 侧边栏导航
selected_exp = st.sidebar.radio("选择实验模块", ['exp1', 'exp2', 'exp3'], format_func=lambda x: PATHS[x]['name'])

# 获取当前实验配置
config = PATHS[selected_exp]

st.header(config['name'])
st.markdown(f"_{config['description']}_")

# 创建三个标签页
tab1, tab2, tab3 = st.tabs(["原始数据", "代码运行与监控", "结果展示"])

with tab1:
    show_file_browser(config['data_dir'], selected_exp)

with tab2:
    st.markdown("### 实时代码执行")
    st.markdown("点击下方按钮，服务器将实时运行 Python 脚本并将日志流式传输到此处。")
    
    col1, col2 = st.columns([1, 3])
    
    script_to_run = col1.radio("选择要运行的脚本:", config['scripts'])
    
    if col1.button(f"运行 {script_to_run}", type="primary"):
        full_script_path = os.path.join(config['program_dir'], script_to_run)
        if os.path.exists(full_script_path):
            run_script_realtime(full_script_path, config['program_dir'])
        else:
            st.error(f"找不到脚本文件: {full_script_path}")

with tab3:
    if st.button("🔄 刷新图库"):
        st.rerun()
    show_gallery(config['plot_dir'])

# 侧边栏额外信息
st.sidebar.markdown("---")
st.sidebar.caption("System Status: Online 🟢")
st.sidebar.caption(f"Root: `{BASE_PATH}`")