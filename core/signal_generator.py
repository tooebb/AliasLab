"""
信号生成模块
支持单频正弦波、双频正弦波、方波、含噪信号
"""

import numpy as np
from scipy.signal import square as scipy_square
from typing import Tuple, Optional


def generate_time_array(duration: float, fs_sim: float) -> np.ndarray:
    """生成高采样率仿真时间轴，用于模拟连续信号

    Args:
        duration: 信号持续时间 (秒)
        fs_sim: 仿真采样率 (Hz)，应远高于信号频率

    Returns:
        等间隔时间数组
    """
    dt = 1.0 / fs_sim
    return np.arange(0, duration, dt)


def single_sine(t: np.ndarray, freq: float, amp: float = 1.0,
                phase: float = 0.0) -> np.ndarray:
    """生成单频正弦波 x(t) = A·sin(2π·f·t + φ)

    Args:
        t: 时间数组
        freq: 频率 (Hz)
        amp: 振幅
        phase: 初始相位 (弧度)

    Returns:
        信号数组
    """
    return amp * np.sin(2 * np.pi * freq * t + phase)


def dual_sine(t: np.ndarray,
              freq1: float, amp1: float,
              freq2: float, amp2: float,
              phase1: float = 0.0, phase2: float = 0.0) -> np.ndarray:
    """生成双频正弦波 x(t) = A1·sin(2π·f1·t) + A2·sin(2π·f2·t)

    Args:
        t: 时间数组
        freq1, freq2: 两个频率分量 (Hz)
        amp1, amp2: 两个振幅
        phase1, phase2: 初始相位

    Returns:
        信号数组
    """
    return (amp1 * np.sin(2 * np.pi * freq1 * t + phase1) +
            amp2 * np.sin(2 * np.pi * freq2 * t + phase2))


def square_wave(t: np.ndarray, freq: float, amp: float = 1.0,
                duty: float = 0.5) -> np.ndarray:
    """生成方波信号

    方波包含基频及奇次谐波 (3f, 5f, 7f, ...)，
    实际带宽远高于基频，采样时需考虑谐波分量。

    Args:
        t: 时间数组
        freq: 基频 (Hz)
        amp: 振幅
        duty: 占空比 (0~1)

    Returns:
        方波信号数组
    """
    return amp * scipy_square(2 * np.pi * freq * t, duty=duty)


def add_noise(signal: np.ndarray, noise_level: float,
              seed: Optional[int] = None) -> np.ndarray:
    """给信号添加高斯白噪声

    Args:
        signal: 原始信号
        noise_level: 噪声标准差
        seed: 随机种子（可复现）

    Returns:
        含噪信号（返回新数组，不修改原数组）
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise


def generate_signal(t: np.ndarray, signal_type: str, **kwargs) -> Tuple[np.ndarray, list]:
    """统一信号生成接口

    Args:
        t: 时间数组
        signal_type: 信号类型
            - 'single_sine':  单频正弦波 (需 freq, 可选 amp, phase)
            - 'dual_sine':    双频正弦波 (需 freq1, freq2, 可选 amp1, amp2)
            - 'square':       方波       (需 freq, 可选 amp, duty)
            - 'noisy_sine':   含噪正弦波 (需 freq, noise_level, 可选 amp, phase)
        **kwargs: 对应类型所需参数

    Returns:
        (signal, freqs): 信号数组和频率分量列表 (Hz)
    """
    if signal_type == 'single_sine':
        freq = kwargs.get('freq', 50)
        amp = kwargs.get('amp', 1.0)
        phase = kwargs.get('phase', 0.0)
        signal = single_sine(t, freq, amp, phase)
        return signal, [freq]

    elif signal_type == 'dual_sine':
        freq1 = kwargs.get('freq1', 40)
        amp1 = kwargs.get('amp1', 1.0)
        freq2 = kwargs.get('freq2', 120)
        amp2 = kwargs.get('amp2', 0.5)
        phase1 = kwargs.get('phase1', 0.0)
        phase2 = kwargs.get('phase2', 0.0)
        signal = dual_sine(t, freq1, amp1, freq2, amp2, phase1, phase2)
        return signal, [freq1, freq2]

    elif signal_type == 'square':
        freq = kwargs.get('freq', 10)
        amp = kwargs.get('amp', 1.0)
        duty = kwargs.get('duty', 0.5)
        signal = square_wave(t, freq, amp, duty)
        # 方波包含基频 + 奇次谐波，返回主要分量（至9次谐波）
        harmonics = [freq * k for k in range(1, 10, 2)]
        return signal, harmonics

    elif signal_type == 'noisy_sine':
        freq = kwargs.get('freq', 50)
        amp = kwargs.get('amp', 1.0)
        phase = kwargs.get('phase', 0.0)
        noise_level = kwargs.get('noise_level', 0.2)
        seed = kwargs.get('seed', 42)
        clean = single_sine(t, freq, amp, phase)
        signal = add_noise(clean, noise_level, seed)
        return signal, [freq]

    else:
        raise ValueError(f"未知的信号类型: {signal_type}")
