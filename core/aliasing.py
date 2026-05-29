"""
混叠分析与频率折叠模块
实现混叠频率计算、频率折叠可视化数据生成、混叠峰检测

奈奎斯特采样定理: fs >= 2·fmax
混叠频率公式: f_alias = |f_signal - k·fs|, k 使得 f_alias ∈ [0, fs/2]
"""

import numpy as np
from typing import List


def compute_alias_freq(f_signal: float, fs: float) -> float:
    """计算单个频率分量的混叠频率

    当 f_signal > fs/2 时，该频率分量在采样后会被折叠到 [0, fs/2] 区间。
    折叠公式: f_alias = |f_signal - k·fs|, 取 k 使结果落在 [0, fs/2]。

    Args:
        f_signal: 原始信号频率 (Hz)
        fs: 采样率 (Hz)

    Returns:
        混叠频率 (Hz)
    """
    if fs <= 0:
        return f_signal

    f_nyquist = fs / 2

    # 已在奈奎斯特范围内，不发生混叠
    if f_signal <= f_nyquist:
        return f_signal

    # 计算最近整数周期的折叠
    k = round(f_signal / fs)
    f_alias = abs(f_signal - k * fs)

    # 确保结果在 [0, fs/2] 内
    if f_alias > f_nyquist:
        f_alias = fs - f_alias

    return f_alias


def compute_all_alias_freqs(signal_freqs: List[float], fs: float) -> List[dict]:
    """计算所有频率分量的混叠频率

    Args:
        signal_freqs: 信号频率分量列表 (Hz)
        fs: 采样率 (Hz)

    Returns:
        每个分量的混叠信息:
        [{'original': 原始频率, 'alias': 混叠后频率, 'shifted': 是否发生折叠}, ...]
    """
    results = []
    for f in signal_freqs:
        f_alias = compute_alias_freq(f, fs)
        results.append({
            'original': f,
            'alias': f_alias,
            'shifted': abs(f - f_alias) > 0.5,  # 允许 0.5Hz 容差
        })
    return results


def check_aliasing(signal_freqs: List[float], fs: float) -> dict:
    """全面检查采样混叠状态

    Args:
        signal_freqs: 信号频率分量列表 (Hz)
        fs: 采样率 (Hz)

    Returns:
        {
            'is_aliasing': bool,         # 是否发生混叠
            'fmax': float,               # 最高频率分量
            'nyquist_rate': float,       # 所需奈奎斯特率 (2*fmax)
            'nyquist_freq': float,       # 奈奎斯特频率 (fs/2)
            'ratio': float,              # 采样比 fs/fmax
            'aliased_freqs': list,       # 发生混叠的分量详情
            'alias_map': dict,           # {原始频率: 混叠频率}
        }
    """
    fmax = max(signal_freqs) if signal_freqs else 0
    nyquist_rate = 2 * fmax
    nyquist_freq = fs / 2
    ratio = fs / fmax if fmax > 0 else float('inf')
    is_aliasing = fs < nyquist_rate

    alias_results = compute_all_alias_freqs(signal_freqs, fs)
    aliased_freqs = [r for r in alias_results if r['shifted']]
    alias_map = {r['original']: r['alias'] for r in aliased_freqs}

    return {
        'is_aliasing': is_aliasing,
        'fmax': fmax,
        'nyquist_rate': nyquist_rate,
        'nyquist_freq': nyquist_freq,
        'ratio': ratio,
        'aliased_freqs': aliased_freqs,
        'alias_map': alias_map,
    }


def generate_spectrum_folding(original_freqs: np.ndarray,
                               original_mag: np.ndarray,
                               fs: float,
                               n_periods: int = 3) -> dict:
    """生成频谱周期延拓数据，用于绘制折叠频谱图

    采样后频谱以 fs 为周期重复。本函数生成多个周期副本，
    用于直观展示频率折叠/混叠的发生机制。

    Args:
        original_freqs: 原始频谱频率轴
        original_mag: 原始频谱幅值
        fs: 采样率
        n_periods: 延拓周期数（左右各 n_periods 个）

    Returns:
        {
            'periods': [(shifted_freqs, mag), ...] 各周期副本,
            'nyquist_freq': fs/2,
            'fs': fs,
        }
    """
    periods = []
    for k in range(-n_periods, n_periods + 1):
        shifted_freqs = original_freqs + k * fs
        periods.append((shifted_freqs, original_mag.copy()))

    return {
        'periods': periods,
        'nyquist_freq': fs / 2,
        'fs': fs,
    }


def detect_alias_peaks(freqs_sampled: np.ndarray,
                        mag_sampled: np.ndarray,
                        signal_freqs: List[float],
                        fs: float,
                        threshold: float = 0.05) -> List[dict]:
    """在采样后频谱中检测混叠导致的异常峰

    对每个信号频率分量，计算其理论混叠频率，然后在实测频谱中
    查找最近的峰值进行验证。

    Args:
        freqs_sampled: 采样后频谱的频率轴
        mag_sampled: 采样后频谱的幅值
        signal_freqs: 原始信号频率分量
        fs: 采样率
        threshold: 峰值检测阈值（相对最大幅值）

    Returns:
        [{'original': f, 'expected_alias': f_alias,
          'detected_peak': f_peak, 'is_aliased': bool, 'peak_match': bool}, ...]
    """
    alias_info = compute_all_alias_freqs(signal_freqs, fs)
    nyquist_freq = fs / 2

    # 简单峰值检测：找局部极大值
    max_mag = np.max(mag_sampled) if len(mag_sampled) > 0 else 1.0
    peaks = []
    for i in range(1, len(mag_sampled) - 1):
        if (mag_sampled[i] > mag_sampled[i - 1] and
            mag_sampled[i] > mag_sampled[i + 1] and
            mag_sampled[i] > threshold * max_mag):
            peaks.append(freqs_sampled[i])

    results = []
    for info in alias_info:
        expected_freq = info['alias']

        # 查找最近的峰值
        if peaks:
            closest_peak = min(peaks, key=lambda p: abs(p - expected_freq))
            distance = abs(closest_peak - expected_freq)
            freq_resolution = nyquist_freq / max(len(mag_sampled) - 1, 1)
            peak_match = distance < max(2.0, freq_resolution * 3)
        else:
            closest_peak = None
            peak_match = False

        results.append({
            'original': info['original'],
            'expected_alias': expected_freq,
            'detected_peak': closest_peak,
            'is_aliased': info['shifted'],
            'peak_match': peak_match,
        })

    return results
