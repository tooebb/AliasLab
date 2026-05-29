"""
可视化模块
使用 Plotly 实现交互式图表，统一深色科技风主题
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, List

# ─── 深色主题配色 ───
C = {
    'bg':         '#0b0f19',
    'paper':      '#111827',
    'grid':       '#1e293b',
    'text':       '#94a3b8',
    'continuous': '#38bdf8',    # 天蓝 — 连续信号
    'samples':    '#f43f5e',    # 玫红 — 采样点
    'recon':      '#f59e0b',    # 琥珀 — 重建信号
    'spectrum':   '#34d399',    # 翠绿 — 频谱填充
    'spec_line':  '#6ee7b7',    # 亮绿 — 频谱线
    'alias':      '#ef4444',    # 红色 — 混叠标注
    'error':      '#f87171',    # 浅红 — 误差
}

# ─── 统一深色 Layout 模板 ───
_DARK = dict(
    template='plotly_dark',
    paper_bgcolor=C['paper'],
    plot_bgcolor=C['bg'],
    font=dict(color=C['text'], size=12),
    height=360,
    margin=dict(l=50, r=25, t=45, b=45),
    hovermode='x unified',
    xaxis=dict(gridcolor=C['grid'], zerolinecolor=C['grid']),
    yaxis=dict(gridcolor=C['grid'], zerolinecolor=C['grid']),
)


def _hex_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ══════════════════════════════════════════════════════════════════════

def plot_time_domain(t: np.ndarray, x: np.ndarray,
                     title: str = "时域波形") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=x, mode='lines',
        line=dict(color=C['continuous'], width=2.2),
        name='连续信号 x(t)',
    ))
    fig.update_layout(title=title, xaxis_title="时间 (s)", yaxis_title="幅值",
                      **_DARK)
    return fig


def plot_sampling(t_cont: np.ndarray, x_cont: np.ndarray,
                  t_sample: np.ndarray, x_sample: np.ndarray,
                  title: str = "采样过程") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_cont, y=x_cont, mode='lines',
        line=dict(color=C['continuous'], width=1.8),
        name='连续信号',
    ))
    fig.add_trace(go.Scatter(
        x=t_sample, y=x_sample, mode='markers',
        marker=dict(color=C['samples'], size=9,
                     symbol='circle-open', line=dict(width=2)),
        name='采样点 x[n]',
    ))
    for tx, xx in zip(t_sample, x_sample):
        fig.add_trace(go.Scatter(
            x=[tx, tx], y=[0, xx], mode='lines',
            line=dict(color=C['samples'], width=0.5, dash='dot'),
            showlegend=False,
        ))
    fig.update_layout(title=title, xaxis_title="时间 (s)", yaxis_title="幅值",
                      **_DARK)
    return fig


def plot_spectrum(freqs: np.ndarray, mag: np.ndarray,
                  title: str = "频谱图",
                  nyquist_freq: Optional[float] = None,
                  alias_annotations: Optional[List[dict]] = None,
                  max_freq: Optional[float] = None,
                  color: str = C['spec_line']) -> go.Figure:
    fig = go.Figure()
    r, g, b = _hex_rgb(C['spectrum'])
    fig.add_trace(go.Scatter(
        x=freqs, y=mag, mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({r},{g},{b},0.18)',
        name='幅度谱',
    ))
    if nyquist_freq is not None:
        fig.add_vline(
            x=nyquist_freq, line_dash="dash",
            line_color=C['alias'], line_width=1.5,
            annotation=dict(
                text=f"  fs/2 = {nyquist_freq:.1f} Hz",
                font=dict(color=C['alias'], size=11),
                showarrow=False, yanchor='top',
            ),
        )
    if alias_annotations:
        for ann in alias_annotations:
            fo = ann.get('original', 0)
            fa = ann.get('alias', 0)
            if abs(fo - fa) < 0.5:
                continue
            yv = np.interp(fa, freqs, mag) if len(freqs) > 0 else 0
            fig.add_annotation(
                x=fa, y=yv,
                text=f"← {fo:.0f} Hz → {fa:.1f} Hz",
                showarrow=True, arrowhead=2, arrowcolor=C['alias'],
                font=dict(color=C['alias'], size=10),
                bgcolor='rgba(0,0,0,0.7)', borderpad=4,
            )
    fig.update_layout(
        title=title,
        xaxis_title="频率 (Hz)", yaxis_title="幅值",
        xaxis_range=[0, max_freq] if max_freq else None,
        **_DARK,
    )
    return fig


def plot_reconstruction(t_original: np.ndarray, x_original: np.ndarray,
                         t_recon: np.ndarray, x_recon: np.ndarray,
                         t_sample: Optional[np.ndarray] = None,
                         x_sample: Optional[np.ndarray] = None,
                         title: str = "信号重建对比") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_original, y=x_original, mode='lines',
        line=dict(color=C['continuous'], width=2.2),
        name='原始信号',
    ))
    fig.add_trace(go.Scatter(
        x=t_recon, y=x_recon, mode='lines',
        line=dict(color=C['recon'], width=2, dash='dash'),
        name='sinc 重建',
    ))
    if t_sample is not None and x_sample is not None:
        fig.add_trace(go.Scatter(
            x=t_sample, y=x_sample, mode='markers',
            marker=dict(color=C['samples'], size=5,
                         symbol='circle-open', line=dict(width=1.2)),
            name='采样点',
        ))
    fig.update_layout(title=title, xaxis_title="时间 (s)", yaxis_title="幅值",
                      **_DARK)
    return fig


def plot_reconstruction_error(t: np.ndarray, x_orig: np.ndarray,
                               x_recon: np.ndarray) -> go.Figure:
    error = x_orig - x_recon
    mse = float(np.mean(error ** 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=error, mode='lines',
        line=dict(color=C['error'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(248,113,113,0.08)',
        name=f'误差 (MSE={mse:.4f})',
    ))
    fig.add_hline(y=0, line_dash="dash", line_color='#4b5563', opacity=0.6)
    fig.update_layout(
        title=f"重建误差  (MSE = {mse:.4f})",
        xaxis_title="时间 (s)", yaxis_title="误差",
        **_DARK,
    )
    return fig


def plot_sampling_gauge(ratio: float, is_aliasing: bool) -> go.Figure:
    if is_aliasing:
        color, label = '#ef4444', '混叠'
    elif ratio < 2.2:
        color, label = '#f59e0b', '临界'
    elif ratio < 5:
        color, label = '#22c55e', '正常'
    else:
        color, label = '#3b82f6', '过采样'

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        title={'text': f"采样状态: {label}", 'font': {'color': color, 'size': 16}},
        number={'suffix': '×', 'font': {'size': 28, 'color': C['text']}},
        delta={'reference': 2.0, 'increasing': {'color': '#22c55e'}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': C['text']},
            'bar': {'color': color, 'thickness': 0.18},
            'bgcolor': C['bg'],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 2], 'color': 'rgba(239,68,68,0.12)'},
                {'range': [2, 4], 'color': 'rgba(245,158,11,0.12)'},
                {'range': [4, 7], 'color': 'rgba(34,197,94,0.12)'},
                {'range': [7, 10], 'color': 'rgba(59,130,246,0.12)'},
            ],
            'threshold': {
                'line': {'color': C['alias'], 'width': 2},
                'thickness': 0.8,
                'value': 2.0,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=C['paper'],
        font=dict(color=C['text']),
        height=220,
        margin=dict(l=30, r=30, t=40, b=10),
    )
    return fig
