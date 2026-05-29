"""
工具函数模块
"""


def get_sampling_status_badge(ratio: float, is_aliasing: bool) -> str:
    """根据采样比返回状态标识文本

    Args:
        ratio: 采样比 fs/fmax
        is_aliasing: 是否发生混叠

    Returns:
        带图标的状态描述字符串
    """
    if is_aliasing:
        if ratio < 1.0:
            return "严重混叠"
        return "混叠"
    elif ratio < 2.2:
        return "临界采样"
    elif ratio < 5:
        return "正常采样"
    else:
        return "深度过采样"


def get_status_color(ratio: float, is_aliasing: bool) -> str:
    """返回采样状态对应的颜色

    Args:
        ratio: 采样比 fs/fmax
        is_aliasing: 是否发生混叠

    Returns:
        CSS 颜色字符串
    """
    if is_aliasing:
        if ratio < 1.0:
            return "#dc3545"  # 红色 - 严重
        return "#fd7e14"      # 橙色 - 混叠
    elif ratio < 2.2:
        return "#ffc107"      # 黄色 - 临界
    elif ratio < 5:
        return "#28a745"      # 绿色 - 正常
    else:
        return "#007bff"      # 蓝色 - 过采样


def format_freq(freq_hz: float) -> str:
    """格式化频率显示，大值自动转换为 kHz

    Args:
        freq_hz: 频率值 (Hz)

    Returns:
        格式化字符串，如 "1.50 kHz" 或 "440.0 Hz"
    """
    if freq_hz >= 1000:
        return f"{freq_hz / 1000:.2f} kHz"
    return f"{freq_hz:.1f} Hz"
