"""
采样模块
实现均匀采样 x[n] = x(n·Ts)，Ts = 1/fs
"""

import numpy as np
from typing import Tuple


def sample_signal(t_continuous: np.ndarray,
                  signal_continuous: np.ndarray,
                  fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """对连续信号进行均匀采样

    使用线性插值从高分辨率仿真信号中提取采样点，
    模拟 ADC 采样过程。

    Args:
        t_continuous: 连续信号时间轴
        signal_continuous: 连续信号值
        fs: 采样率 (Hz)

    Returns:
        (t_sample, x_sample): 采样时间点和采样值
    """
    Ts = 1.0 / fs
    duration = t_continuous[-1] - t_continuous[0]
    n_samples = int(np.floor(duration * fs))
    n_samples = max(n_samples, 2)

    t_sample = np.arange(n_samples) * Ts + t_continuous[0]
    x_sample = np.interp(t_sample, t_continuous, signal_continuous)

    return t_sample, x_sample


def get_nyquist_rate(fmax: float) -> float:
    """计算奈奎斯特采样率 R_nyquist = 2·fmax

    Args:
        fmax: 信号最高频率分量 (Hz)

    Returns:
        奈奎斯特采样率 (Hz)
    """
    return 2.0 * fmax


def get_sampling_ratio(fs: float, fmax: float) -> float:
    """计算采样率与最高频率的比值

    Args:
        fs: 采样率 (Hz)
        fmax: 信号最高频率分量 (Hz)

    Returns:
        fs / fmax 比值
    """
    return fs / fmax if fmax > 0 else float('inf')
