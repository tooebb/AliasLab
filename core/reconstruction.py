"""
信号重建模块
使用 sinc 插值从采样点重建连续信号

重建公式: x(t) = Σ x[n] · sinc((t - n·Ts) / Ts)
其中 sinc(x) = sin(πx) / (πx)
"""

import numpy as np


def sinc_interp(x: np.ndarray) -> np.ndarray:
    """归一化 sinc 函数: sinc(x) = sin(πx) / (πx)

    在 x=0 处取极限值 1。

    Args:
        x: 输入数组

    Returns:
        sinc(x) 值
    """
    result = np.ones_like(x)
    nonzero = x != 0
    result[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    return result


def reconstruct_signal(t_sample: np.ndarray,
                        x_sample: np.ndarray,
                        t_reconstruct: np.ndarray,
                        fs: float,
                        window_half: int = 50) -> np.ndarray:
    """使用 sinc 插值从采样点重建连续信号

    理论上需要无限个采样点进行 sinc 插值，实际计算中使用有限窗口
    (±window_half 个采样点) 截断以兼顾效率与精度。

    Args:
        t_sample: 采样时间点
        x_sample: 采样值 x[n]
        t_reconstruct: 待重建的时间轴
        fs: 采样率 (Hz)
        window_half: 插值窗口半宽（采样点数），越大越精确但越慢

    Returns:
        重建信号数组
    """
    Ts = 1.0 / fs
    x_reconstructed = np.zeros_like(t_reconstruct)
    n_samples = len(t_sample)

    for i, t in enumerate(t_reconstruct):
        # 找到最近的采样点索引
        center_idx = int(round((t - t_sample[0]) / Ts))
        start = max(0, center_idx - window_half)
        end = min(n_samples, center_idx + window_half + 1)

        # sinc 插值求和
        for j in range(start, end):
            x_reconstructed[i] += x_sample[j] * sinc_interp(
                (t - t_sample[j]) / Ts
            )

    return x_reconstructed


def compute_reconstruction_error(x_original: np.ndarray,
                                  x_reconstructed: np.ndarray) -> dict:
    """计算重建误差指标

    Args:
        x_original: 原始信号
        x_reconstructed: 重建信号

    Returns:
        {'mse': 均方误差, 'mae': 平均绝对误差, 'max_error': 最大误差}
    """
    error = x_original - x_reconstructed
    return {
        'mse': float(np.mean(error ** 2)),
        'mae': float(np.mean(np.abs(error))),
        'max_error': float(np.max(np.abs(error))),
    }
