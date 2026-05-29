"""
可视化模块
使用 Plotly 实现交互式图表，支持混叠标注、频谱对比、重建分析
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, List

# ─── 全局配色方案 ───
COLORS = {
    'continuous':    '#1f77b4',   # 蓝色 — 连续信号
    'samples':       '#d62728',   # 红色 — 采样点
    'reconstructed': '#ff7f0e',   # 橙色 — 重建信号
    'spectrum':      '#2ca02c',   # 绿色 — 频谱
    'alias':         '#d62728',   # 红色 — 混叠标注线
    'alias_dash':    '#9467bd',   # 紫色 — 混叠辅助线
    'error':         '#e74c3c',   # 红色 — 误差
}

_LAYOUT_BASE = dict(
    template='plotly_white',
    height=380,
    margin=dict(l=50, r=30, t=50, b=50),
    hovermode='x unified',
)


# ══════════════════════════════════════════════════════════════════════
# 时域波形
# ══════════════════════════════════════════════════════════════════════

def plot_time_domain(t: np.ndarray, x: np.ndarray,
                     title: str = "时域波形") -> go.Figure:
    """绘制连续信号时域波形

    Args:
        t: 时间轴
        x: 信号值
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=x, mode='lines',
        line=dict(color=COLORS['continuous'], width=1.8),
        name='连续信号 x(t)',
    ))
    fig.update_layout(title=title, xaxis_title="时间 (s)", yaxis_title="幅值",
                      **_LAYOUT_BASE)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 采样过程
# ══════════════════════════════════════════════════════════════════════

def plot_sampling(t_cont: np.ndarray, x_cont: np.ndarray,
                  t_sample: np.ndarray, x_sample: np.ndarray,
                  title: str = "采样过程") -> go.Figure:
    """绘制连续信号叠加采样点

    Args:
        t_cont: 连续信号时间轴
        x_cont: 连续信号值
        t_sample: 采样时间点
        x_sample: 采样值
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    # 连续信号（浅色背景线）
    fig.add_trace(go.Scatter(
        x=t_cont, y=x_cont, mode='lines',
        line=dict(color=COLORS['continuous'], width=1.8),
        name='原始连续信号',
    ))

    # 采样点
    fig.add_trace(go.Scatter(
        x=t_sample, y=x_sample, mode='markers',
        marker=dict(color=COLORS['samples'], size=10,
                     symbol='circle-open', line=dict(width=2)),
        name='采样点 x[n]',
    ))

    # stem 线（采样点 → 零轴）
    for tx, xx in zip(t_sample, x_sample):
        fig.add_trace(go.Scatter(
            x=[tx, tx], y=[0, xx], mode='lines',
            line=dict(color=COLORS['samples'], width=0.6, dash='dot'),
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间 (s)", yaxis_title="幅值",
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# 频谱图（含混叠标注）
# ══════════════════════════════════════════════════════════════════════

def plot_spectrum(freqs: np.ndarray, mag: np.ndarray,
                  title: str = "频谱图",
                  nyquist_freq: Optional[float] = None,
                  alias_annotations: Optional[List[dict]] = None,
                  max_freq: Optional[float] = None,
                  color: str = COLORS['spectrum']) -> go.Figure:
    """绘制单边幅度谱，支持混叠标注

    Args:
        freqs: 频率轴 (Hz)
        mag: 幅值
        title: 图表标题
        nyquist_freq: 奈奎斯特频率 fs/2（标红色虚线）
        alias_annotations: 混叠标注列表 [{'original': f0, 'alias': fa}, ...]
        max_freq: x 轴频率上限
        color: 频谱曲线颜色

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    # 频谱曲线 + 填充
    fig.add_trace(go.Scatter(
        x=freqs, y=mag, mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba{(*_hex_to_rgb(color), 0.15)}',
        name='幅度谱',
    ))

    # 奈奎斯特频率参考线
    if nyquist_freq is not None:
        fig.add_vline(
            x=nyquist_freq, line_dash="dash",
            line_color=COLORS['alias'], line_width=1.5,
            annotation=dict(
                text=f"  fs/2 = {nyquist_freq:.1f} Hz",
                font=dict(color=COLORS['alias'], size=11),
                showarrow=False, yanchor='top',
            ),
        )

    # 混叠频率标注
    if alias_annotations:
        for ann in alias_annotations:
            f_orig = ann.get('original', 0)
            f_alias = ann.get('alias', 0)
            if abs(f_orig - f_alias) < 0.5:
                continue  # 未发生混叠，不标注

            # 在 aliased 位置取幅值用于标注箭头
            y_at_alias = np.interp(f_alias, freqs, mag) if len(freqs) > 0 else 0
            fig.add_annotation(
                x=f_alias, y=y_at_alias,
                text=f"← {f_orig:.0f} Hz 混叠至 {f_alias:.1f} Hz",
                showarrow=True, arrowhead=2, arrowcolor=COLORS['alias'],
                font=dict(color=COLORS['alias'], size=10),
                bgcolor='rgba(255,255,255,0.85)',
                borderpad=4,
            )

    fig.update_layout(
        title=title,
        xaxis_title="频率 (Hz)", yaxis_title="幅值",
        xaxis_range=[0, max_freq] if max_freq else None,
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# 频谱对比（原始 vs 采样后，上下并排）
# ══════════════════════════════════════════════════════════════════════

def plot_spectrum_comparison(freqs_orig: np.ndarray, mag_orig: np.ndarray,
                              freqs_sampled: np.ndarray, mag_sampled: np.ndarray,
                              nyquist_freq: Optional[float] = None,
                              alias_annotations: Optional[List[dict]] = None,
                              max_freq: Optional[float] = None) -> go.Figure:
    """上下并排对比原始频谱 vs 采样后频谱

    Args:
        freqs_orig: 原始频谱频率轴
        mag_orig: 原始频谱幅值
        freqs_sampled: 采样后频谱频率轴
        mag_sampled: 采样后频谱幅值
        nyquist_freq: 奈奎斯特频率参考线
        alias_annotations: 混叠标注
        max_freq: x 轴频率上限

    Returns:
        Plotly Figure 对象（双子图）
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("原始信号频谱（高采样率仿真）", "采样后频谱"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=freqs_orig, y=mag_orig, mode='lines',
        line=dict(color=COLORS['spectrum'], width=1.8),
        fill='tozeroy',
        fillcolor=f'rgba{(*_hex_to_rgb(COLORS["spectrum"]), 0.15)}',
        name='原始频谱',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_sampled, y=mag_sampled, mode='lines',
        line=dict(color=COLORS['continuous'], width=2),
        fill='tozeroy',
        fillcolor=f'rgba{(*_hex_to_rgb(COLORS["continuous"]), 0.15)}',
        name='采样后频谱',
    ), row=2, col=1)

    if nyquist_freq is not None:
        for r in [1, 2]:
            fig.add_vline(
                x=nyquist_freq, line_dash="dash",
                line_color=COLORS['alias'], line_width=1.5,
                row=r, col=1,
            )

    fig.update_layout(
        title="频谱对比分析",
        xaxis2_title="频率 (Hz)",
        yaxis_title="幅值",
        xaxis_range=[0, max_freq] if max_freq else None,
        height=520,
        margin=dict(l=50, r=30, t=60, b=50),
        template='plotly_white',
        hovermode='x unified',
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# 信号重建对比
# ══════════════════════════════════════════════════════════════════════

def plot_reconstruction(t_original: np.ndarray, x_original: np.ndarray,
                         t_recon: np.ndarray, x_recon: np.ndarray,
                         t_sample: Optional[np.ndarray] = None,
                         x_sample: Optional[np.ndarray] = None,
                         title: str = "信号重建对比") -> go.Figure:
    """原始信号 vs sinc 重建信号叠加对比

    Args:
        t_original: 原始信号时间轴
        x_original: 原始信号值
        t_recon: 重建信号时间轴
        x_recon: 重建信号值
        t_sample: 采样时间点（可选，用于叠加显示）
        x_sample: 采样值（可选）
        title: 图表标题

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_original, y=x_original, mode='lines',
        line=dict(color=COLORS['continuous'], width=2),
        name='原始信号',
    ))

    fig.add_trace(go.Scatter(
        x=t_recon, y=x_recon, mode='lines',
        line=dict(color=COLORS['reconstructed'], width=2, dash='dash'),
        name='sinc 重建信号',
    ))

    if t_sample is not None and x_sample is not None:
        fig.add_trace(go.Scatter(
            x=t_sample, y=x_sample, mode='markers',
            marker=dict(color=COLORS['samples'], size=6,
                         symbol='circle-open', line=dict(width=1.5)),
            name='采样点',
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间 (s)", yaxis_title="幅值",
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# 重建误差曲线
# ══════════════════════════════════════════════════════════════════════

def plot_reconstruction_error(t: np.ndarray, x_orig: np.ndarray,
                               x_recon: np.ndarray) -> go.Figure:
    """绘制重建误差曲线 e(t) = x_orig(t) - x_recon(t)

    Args:
        t: 时间轴
        x_orig: 原始信号
        x_recon: 重建信号

    Returns:
        Plotly Figure 对象
    """
    error = x_orig - x_recon
    mse = np.mean(error ** 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=error, mode='lines',
        line=dict(color=COLORS['error'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.08)',
        name=f'误差 (MSE={mse:.4f})',
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"重建误差曲线 (MSE = {mse:.4f})",
        xaxis_title="时间 (s)", yaxis_title="误差",
        **_LAYOUT_BASE,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# 采样状态仪表盘
# ══════════════════════════════════════════════════════════════════════

def plot_sampling_gauge(ratio: float, is_aliasing: bool) -> go.Figure:
    """绘制采样状态仪表盘

    Args:
        ratio: 采样比 fs/fmax
        is_aliasing: 是否混叠

    Returns:
        Plotly Figure 对象（小尺寸仪表盘）
    """
    if is_aliasing:
        color = '#dc3545'
        label = '混叠'
    elif ratio < 2.2:
        color = '#ffc107'
        label = '临界'
    elif ratio < 5:
        color = '#28a745'
        label = '正常'
    else:
        color = '#007bff'
        label = '过采样'

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        title={'text': f"采样状态: {label}", 'font': {'color': color, 'size': 16}},
        number={'suffix': '×', 'font': {'size': 28}},
        delta={'reference': 2.0, 'increasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.2},
            'steps': [
                {'range': [0, 2], 'color': 'rgba(220,53,69,0.15)'},
                {'range': [2, 4], 'color': 'rgba(255,193,7,0.15)'},
                {'range': [4, 7], 'color': 'rgba(40,167,69,0.15)'},
                {'range': [7, 10], 'color': 'rgba(0,123,255,0.15)'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 2},
                'thickness': 0.8,
                'value': 2.0,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=30, r=30, t=40, b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> tuple:
    """将 #rrggbb 转为 (r, g, b) 元组，值域 0-255"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
