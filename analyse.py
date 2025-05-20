import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
import datetime

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 单独定义波峰和波谷检测函数
def find_peaks_adaptive(signal_data, peak_type='peak'):
    """自适应波峰/波谷检测
    peak_type: 'peak'检测波峰, 'valley'检测波谷
    """
    if peak_type == 'valley':
        signal_data = -signal_data  # 波谷检测
    
    mean_value = np.mean(signal_data)
    std_dev = np.std(signal_data)
    # 计算阈值
    threshold = mean_value + 2 * std_dev
    
    # 预检测为参数优化
    peaks_pre, _ = signal.find_peaks(signal_data, height=threshold)
    
    # 智能设置distance参数
    if len(peaks_pre) > 1:
        peak_intervals_pre = np.diff(peaks_pre)
        distance = max(int(np.median(peak_intervals_pre) * 0.5), 5)  # 至少5点间隔
    else:
        distance = 5  # 默认最小间隔
    
    # 智能设置prominence参数
    prominence = max(1.0 * std_dev, 0.1 * (np.max(signal_data) - np.min(signal_data)))
    
    # 最终检测
    peaks, properties = signal.find_peaks(signal_data, height=threshold, 
                                          distance=distance, prominence=prominence)
    
    return peaks, properties

# 主分析函数
def analyze_neural_activity(signal_data, fs, key_name, output_dir=None):
    """完整分析神经放电数据"""
    # 基本统计分析
    mean_value = np.mean(signal_data)
    std_dev = np.std(signal_data)
    max_value = np.max(signal_data)
    min_value = np.min(signal_data)
    
    # 检测波峰和波谷
    peaks, peak_props = find_peaks_adaptive(signal_data, peak_type='peak')
    valleys, valley_props = find_peaks_adaptive(signal_data, peak_type='valley')
    
    # 计算波峰/波谷间隔和平均值
    peak_intervals = np.diff(peaks) / fs if len(peaks) > 1 else [0]
    valley_intervals = np.diff(valleys) / fs if len(valleys) > 1 else [0]
    
    mean_peak_interval = np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
    mean_valley_interval = np.mean(valley_intervals) if len(valley_intervals) > 0 else 0
    
    peak_values = signal_data[peaks]
    valley_values = signal_data[valleys]
    
    average_peak_value = np.mean(peak_values) if len(peak_values) > 0 else 0
    average_valley_value = np.mean(valley_values) if len(valley_values) > 0 else 0
    
    # 计算放电频率 (Hz)
    discharge_rate = 1/mean_peak_interval if mean_peak_interval > 0 else 0
    
    # 绘制信号与波峰波谷
    times = np.arange(len(signal_data)) / fs
    plt.figure(figsize=(18, 6))
    plt.plot(times, signal_data, label="信号")
    plt.plot(times[valleys], valley_values, 'go', label="波谷")
    plt.plot(times[peaks], peak_values, 'ro', label="波峰")
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅值 (mV)')
    plt.legend()
    plt.title(f'信号波形与放电分析 - {key_name}')
    
    # 统计信息文本
    stats_text = (
        f"均值: {mean_value:.4g} mV    标准差: {std_dev:.4g} mV    "
        f"最大值: {max_value:.4g} mV    最小值: {min_value:.4g} mV\n"
        f"波峰平均间隔: {mean_peak_interval:.4g} 秒    波峰平均值: {average_peak_value:.4g} mV    放电频率: {discharge_rate:.2f} Hz\n"
        f"波谷平均间隔: {mean_valley_interval:.4g} 秒    波谷平均值: {average_valley_value:.4g} mV"
    )
    
    plt.figtext(0.5, -0.05, stats_text, ha='center', va='top', fontsize=12)
    plt.tight_layout()
    
    # 导出图像
    if output_dir:
        waveform_dir = os.path.join(output_dir, "波形图")
        os.makedirs(waveform_dir, exist_ok=True)
        plt.savefig(os.path.join(waveform_dir, f"{key_name}_waveform.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        'key': key_name,
        'mean': float(mean_value), 
        'std': float(std_dev),
        'max': float(max_value),
        'min': float(min_value),
        'discharge_rate': float(discharge_rate),
        'peak_count': int(len(peaks)),
        'valley_count': int(len(valleys)),
        'mean_peak_interval': float(mean_peak_interval),
        'mean_valley_interval': float(mean_valley_interval),
        'average_peak_value': float(average_peak_value),
        'average_valley_value': float(average_valley_value)
    }

# 频谱分析函数
def spectral_analysis(signal_data, fs, key_name, freq_min=1, freq_max=50, output_dir=None):
    """频谱分析与可视化"""
    # 智能选择NFFT
    signal_len = len(signal_data)
    power_of_2 = int(np.floor(np.log2(signal_len)))
    NFFT = 2 ** max(power_of_2-5, 8)  # 至少256点FFT
    
    noverlap = int(NFFT / 2)
    
    plt.figure(figsize=(12, 8))
    
    # 计算频谱
    Pxx, freqs, bins, _ = plt.specgram(signal_data, Fs=fs, NFFT=NFFT, 
                                      noverlap=noverlap, cmap='jet')
    plt.clf()  # 清除初始图
    
    # 频率范围过滤
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    
    # 绘制频谱图
    plt.subplot(211)
    plt.pcolormesh(bins, freqs[freq_mask], 10 * np.log10(Pxx[freq_mask, :]), 
                  cmap='jet', shading='auto')
    plt.colorbar(label='功率/频率 (dB/Hz)')
    plt.ylabel('频率 (Hz)')
    plt.title(f'时频分析 ({freq_min}-{freq_max} Hz) - {key_name}')
    
    # 添加平均功率谱密度图
    plt.subplot(212)
    mean_psd = np.mean(Pxx[freq_mask, :], axis=1)
    plt.semilogy(freqs[freq_mask], mean_psd)
    plt.grid(True)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度 (V²/Hz)')
    plt.title(f'平均功率谱密度 - {key_name}')
    
    # 标记显著频率
    peak_indices = signal.find_peaks(mean_psd, prominence=np.max(mean_psd)*0.1)[0]
    significant_freqs = freqs[freq_mask][peak_indices]
    
    # 保存频率数据
    top_freq_values = []
    top_freq_powers = []
    
    if len(significant_freqs) > 0:
        # 按照功率从高到低排序频率
        sorted_indices = np.argsort(mean_psd[peak_indices])[::-1]
        sorted_freqs = significant_freqs[sorted_indices]
        sorted_powers = mean_psd[peak_indices][sorted_indices]
        
        # 标记前3个最显著频率
        for i, (freq, power) in enumerate(zip(sorted_freqs[:3], sorted_powers[:3])):
            if freq > freq_min:  # 忽略接近0的部分
                plt.axvline(x=freq, color='r', linestyle='--', alpha=0.5)
                plt.text(freq, np.max(mean_psd), f'{freq:.1f}Hz', 
                        rotation=90, verticalalignment='top')
                top_freq_values.append(float(freq))
                top_freq_powers.append(float(power))
    
    # 补齐3个频率值(不足的用0填充)
    while len(top_freq_values) < 3:
        top_freq_values.append(0)
        top_freq_powers.append(0)
    
    plt.tight_layout()
    
    # 导出图像
    if output_dir:
        spectrum_dir = os.path.join(output_dir, "频谱图")
        os.makedirs(spectrum_dir, exist_ok=True)
        plt.savefig(os.path.join(spectrum_dir, f"{key_name}_spectrum.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        'key': key_name,
        'dominant_freq1': top_freq_values[0] if len(top_freq_values) > 0 else 0,
        'dominant_power1': top_freq_powers[0] if len(top_freq_powers) > 0 else 0,
        'dominant_freq2': top_freq_values[1] if len(top_freq_values) > 1 else 0,
        'dominant_power2': top_freq_powers[1] if len(top_freq_powers) > 1 else 0,
        'dominant_freq3': top_freq_values[2] if len(top_freq_values) > 2 else 0,
        'dominant_power3': top_freq_powers[2] if len(top_freq_powers) > 2 else 0,
    }

# 能量密度分析函数
def energy_density_analysis(signal_data, fs, key_name, window_size=None, overlap=0.5, output_dir=None):
    """信号能量密度分析"""
    # 自适应窗口大小
    if window_size is None or window_size <= 0:
        signal_len = len(signal_data)
        window_size = min(2**int(np.log2(signal_len/20)), signal_len//10)
        window_size = max(window_size, 256)  # 最小窗口大小
    
    step_size = int(window_size * (1 - overlap))
    
    # 计算能量
    energies = []
    time_points = []
    
    for start in range(0, len(signal_data) - window_size, step_size):
        end = start + window_size
        window_data = signal_data[start:end]
        
        # 能量计算 (可选用不同方法)
        energy = np.sum(np.square(window_data - np.mean(window_data)))  # 去均值后平方和
        
        energies.append(energy)
        time_points.append((start + window_size/2) / fs)  # 窗口中心点时间
    
    # 转换为numpy数组
    energies = np.array(energies)
    time_points = np.array(time_points)
    
    # 绘制能量密度图
    plt.figure(figsize=(12, 8))
    
    # 线性尺度
    plt.subplot(211)
    plt.plot(time_points, energies, label="能量密度")
    plt.grid(True)
    plt.xlabel('时间 (秒)')
    plt.ylabel('能量密度')
    plt.title(f'信号能量密度 (线性尺度) - {key_name}')
    
    # 对数尺度
    plt.subplot(212)
    log_energies = 10 * np.log10(energies + 1e-6)  # 防止log(0)
    plt.plot(time_points, log_energies, label="能量密度(dB)", color='r')
    plt.grid(True)
    plt.xlabel('时间 (秒)')
    plt.ylabel('能量密度 (dB)')
    plt.title(f'信号能量密度 (对数尺度) - {key_name}')
    
    # 检测能量突变
    energy_mean = np.mean(log_energies)
    energy_std = np.std(log_energies)
    energy_threshold = energy_mean + 2*energy_std
    
    high_energy_points = time_points[log_energies > energy_threshold]
    high_energy_values = log_energies[log_energies > energy_threshold]
    
    if len(high_energy_points) > 0:
        plt.scatter(high_energy_points, high_energy_values, color='g', marker='o')
    
    plt.tight_layout()
    
    # 导出图像
    if output_dir:
        energy_dir = os.path.join(output_dir, "能量图")
        os.makedirs(energy_dir, exist_ok=True)
        plt.savefig(os.path.join(energy_dir, f"{key_name}_energy.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 计算能量相关特征
    mean_energy = np.mean(energies)
    max_energy = np.max(energies)
    std_energy = np.std(energies)
    energy_variation = std_energy / mean_energy if mean_energy > 0 else 0
    
    # 计算高能量事件百分比
    high_energy_percentage = len(high_energy_points) / len(log_energies) * 100 if len(log_energies) > 0 else 0
    
    return {
        'key': key_name,
        'mean_energy': float(mean_energy),
        'max_energy': float(max_energy),
        'std_energy': float(std_energy),
        'energy_variation': float(energy_variation),
        'high_energy_count': int(len(high_energy_points)),
        'high_energy_percentage': float(high_energy_percentage)
    }

# 批量处理函数
def batch_process_data(npz_file_path, output_dir, key_list=None, fs=1000):
    """
    批量处理数据集中的多个键值
    
    参数:
    - npz_file_path: .npz文件路径
    - output_dir: 输出目录
    - key_list: 要处理的键列表，如果为None则处理所有键
    - fs: 采样率，默认1000Hz
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据
    try:
        mat_data = np.load(npz_file_path, allow_pickle=True)
        print(f"成功加载数据文件: {npz_file_path}")
        print(f"数据集包含键: {mat_data.files}")
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return
    
    # 确定要处理的键
    if key_list is None:
        key_list = mat_data.files
    else:
        # 只保留数据中存在的键
        key_list = [key for key in key_list if key in mat_data.files]
    
    if not key_list:
        print("没有找到要处理的有效键!")
        return
    
    print(f"将处理以下键: {key_list}")
    
    # 存储结果的列表
    waveform_results = []
    spectral_results = []
    energy_results = []
    
    # 处理每个键
    total_keys = len(key_list)
    for i, key in enumerate(key_list):
        print(f"处理 {key} ({i+1}/{total_keys})...")
        
        try:
            # 获取数据并确保它是一维数组
            signal_data = mat_data[key].flatten()
            
            # 执行三种分析
            wave_result = analyze_neural_activity(signal_data, fs, key, output_dir)
            spec_result = spectral_analysis(signal_data, fs, key, output_dir=output_dir)
            energy_result = energy_density_analysis(signal_data, fs, key, output_dir=output_dir)
            
            # 添加结果到列表
            waveform_results.append(wave_result)
            spectral_results.append(spec_result)
            energy_results.append(energy_result)
            
            print(f"  完成 {key} 的分析")
        except Exception as e:
            print(f"  处理 {key} 时出错: {e}")
    
    # 创建数据帧并导出到CSV
    if waveform_results:
        waveform_df = pd.DataFrame(waveform_results)
        spectral_df = pd.DataFrame(spectral_results)
        energy_df = pd.DataFrame(energy_results)
        
        # 合并所有数据帧
        merged_df = pd.merge(waveform_df, spectral_df, on='key')
        merged_df = pd.merge(merged_df, energy_df, on='key')
        
        # 创建结果CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"analysis_results_{timestamp}.csv")
        merged_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"分析结果已保存到: {csv_path}")
    else:
        print("没有生成任何分析结果")
    
    return {
        'waveform': waveform_results,
        'spectral': spectral_results,
        'energy': energy_results
    }

# 主执行代码
if __name__ == "__main__":
    # 文件路径设置
    npz_file_path = r"D:\神经生物学\20250513核团放电\nuc_v2_split.npz"  # 修改为你的数据文件路径
    output_base_dir = r"D:\神经生物学\分析结果"  # 修改为你想要的输出目录
    
    # 创建带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"analysis_{timestamp}")
    
    # 采样率设置
    fs = 1000  # 根据你的数据修改
    
    # 要处理的键列表，如果为None则处理所有键
    # key_list = ['7900', '8000', '8100']  # 示例：只处理这些键
    key_list = None  # 处理所有键
    
    # 执行批量处理
    results = batch_process_data(npz_file_path, output_dir, key_list, fs)
    
    print("处理完成!")
