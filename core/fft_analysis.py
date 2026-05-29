"""
FFT 频谱分析模块
计算并返回单边幅度谱
"""

import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple


def compute_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """计算信号的单边幅度谱

    使用 scipy.fft.fft 计算 DFT，返回正频率部分的幅值谱。

    Args:
        signal: 输入信号 (1D 数组)
        fs: 采样率 (Hz)

    Returns:
        (freqs, magnitude): 正频率轴 (Hz) 和对应的幅值
    """
    n = len(signal)
    yf = fft(signal)
    freqs = fftfreq(n, 1.0 / fs)

    # 取正频率部分 (包含 DC)
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    magnitude = (2.0 / n) * np.abs(yf[positive_mask])

    # DC 分量不需要乘 2
    if len(magnitude) > 0:
        magnitude[0] /= 2

    return freqs, magnitude


def find_spectral_peaks(freqs: np.ndarray, mag: np.ndarray,
                         threshold: float = 0.05,
                         min_distance: float = 1.0) -> list:
    """在频谱中查找峰值频率

    Args:
        freqs: 频率轴
        mag: 幅值
        threshold: 相对最大值的阈值 (0~1)
        min_distance: 峰值间最小频率间隔 (Hz)

    Returns:
        [(freq, mag), ...] 峰值列表，按幅值降序排列
    """
    if len(mag) < 3:
        return []

    max_mag = np.max(mag)
    if max_mag == 0:
        return []

    peaks = []
    for i in range(1, len(mag) - 1):
        if (mag[i] > mag[i - 1] and
            mag[i] > mag[i + 1] and
            mag[i] > threshold * max_mag):
            peaks.append((freqs[i], mag[i]))

    # 按幅值降序，移除过近的峰
    peaks.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for f, m in peaks:
        if all(abs(f - pf) > min_distance for pf, _ in filtered):
            filtered.append((f, m))

    return filtered
