"""
AliasLab — 采样混叠与重建智能仿真系统
========================================
面向数字信号处理教学的采样-混叠-重建全过程智能分析平台
"""

import streamlit as st
import numpy as np
import time

from core.signal_generator import generate_time_array, generate_signal
from core.sampling import sample_signal
from core.aliasing import check_aliasing
from core.fft_analysis import compute_spectrum
from core.reconstruction import reconstruct_signal, compute_reconstruction_error
from agent.sampling_agent import SamplingAgent
from utils.plotting import (
    plot_time_domain, plot_sampling, plot_spectrum,
    plot_reconstruction, plot_reconstruction_error, plot_sampling_gauge,
)
from utils.helpers import get_sampling_status_badge, get_status_color

# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AliasLab — DSP 智能实验平台",
    page_icon="",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════
# 深色主题 CSS
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── 全局底色 ── */
    .stApp {
        background: #0b0f19;
    }
    .stMain {
        background: #0b0f19;
    }

    /* ── 顶部 Header（Streamlit 自带）透明化 ── */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ── 侧边栏: 半透明玻璃感 ── */
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.92);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(56, 189, 248, 0.1) !important;
    }

    /* ── 标题发光 ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0.2rem 0 0 0;
        letter-spacing: 2px;
    }

    /* ── KPI 卡片 ── */
    .kpi-row {
        display: flex;
        gap: 12px;
        margin: 1rem 0;
    }
    .kpi-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        transition: border-color 0.3s;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .kpi-unit {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* ── 区块卡片 ── */
    .section-card {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-card h3, .section-card h4 {
        color: #e2e8f0 !important;
        margin-top: 0;
    }

    /* ── Agent 分析面板 ── */
    .agent-panel {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
    }
    .agent-panel .agent-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(167, 139, 250, 0.15);
    }

    /* ── 状态脉冲灯 ── */
    @keyframes pulse-dot {
        0%, 100% { box-shadow: 0 0 4px currentColor; }
        50% { box-shadow: 0 0 14px currentColor, 0 0 28px currentColor; }
    }
    .status-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 2s infinite;
    }

    /* ── 按钮 ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        background: rgba(56, 189, 248, 0.08) !important;
        color: #38bdf8 !important;
        transition: all 0.25s;
    }
    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.18) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
    }

    /* ── Metric ── */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 0.3rem 0.6rem;
        border: 1px solid rgba(56, 189, 248, 0.06);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #64748b !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        background: rgba(56, 189, 248, 0.06);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        color: #94a3b8 !important;
        border-radius: 10px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
FS_SIM = 5000
MAX_DISPLAY_FREQ = 300

if 'agent' not in st.session_state:
    st.session_state.agent = SamplingAgent()
agent = st.session_state.agent

# ══════════════════════════════════════════════════════════════════════
# 发光标题
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:0.6rem 0 0.2rem 0;">
    <h1 class="hero-title">AliasLab</h1>
    <p class="hero-subtitle">采样混叠与重建智能仿真系统&nbsp;&nbsp;·&nbsp;&nbsp;DSP INTELLIGENT EXPERIMENT PLATFORM</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# 侧边栏 —— 控制中心
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 控制中心")

    st.markdown("##### 信号源")
    signal_type = st.selectbox(
        "类型",
        ["单频正弦波", "双频正弦波", "方波", "含噪正弦波"],
        label_visibility="collapsed",
    )
    duration = st.slider("时长 (s)", 0.1, 2.0, 0.5, 0.05)

    noise_level = 0.0
    if signal_type == "单频正弦波":
        freq1 = st.slider("频率 f (Hz)", 1, 300, 50, 1)
        amp1 = st.slider("振幅 A", 0.1, 5.0, 1.0, 0.1)
        freq2, amp2 = None, None
    elif signal_type == "双频正弦波":
        c1, c2 = st.columns(2)
        with c1:
            freq1 = st.slider("f₁ (Hz)", 1, 300, 40, 1)
            amp1 = st.slider("A₁", 0.1, 5.0, 1.0, 0.1)
        with c2:
            freq2 = st.slider("f₂ (Hz)", 1, 300, 150, 1)
            amp2 = st.slider("A₂", 0.1, 5.0, 0.5, 0.1)
        st.caption("设 f₂ > fs/2 触发混叠")
    elif signal_type == "方波":
        freq1 = st.slider("基频 f (Hz)", 1, 100, 10, 1)
        amp1 = st.slider("振幅 A", 0.1, 5.0, 1.0, 0.1)
        freq2, amp2 = None, None
        st.info("含奇次谐波至 9f，带宽远高于基频")
    elif signal_type == "含噪正弦波":
        freq1 = st.slider("频率 f (Hz)", 1, 300, 50, 1)
        amp1 = st.slider("振幅 A", 0.1, 5.0, 1.0, 0.1)
        noise_level = st.slider("噪声 σ", 0.0, 1.0, 0.2, 0.05)
        freq2, amp2 = None, None

    st.markdown("---")
    st.markdown("##### 采样器")
    fs = st.slider("采样率 fs (Hz)", 1, 600, 200, 1)

# ══════════════════════════════════════════════════════════════════════
# 信号生成
# ══════════════════════════════════════════════════════════════════════

t_continuous = generate_time_array(duration, FS_SIM)

if signal_type == "单频正弦波":
    x_continuous, signal_freqs = generate_signal(t_continuous, 'single_sine', freq=freq1, amp=amp1)
elif signal_type == "双频正弦波":
    x_continuous, signal_freqs = generate_signal(t_continuous, 'dual_sine', freq1=freq1, amp1=amp1, freq2=freq2, amp2=amp2)
elif signal_type == "方波":
    x_continuous, signal_freqs = generate_signal(t_continuous, 'square', freq=freq1, amp=amp1)
elif signal_type == "含噪正弦波":
    x_continuous, signal_freqs = generate_signal(t_continuous, 'noisy_sine', freq=freq1, amp=amp1, noise_level=noise_level)

t_sample, x_sample = sample_signal(t_continuous, x_continuous, fs)
freqs_orig, mag_orig = compute_spectrum(x_continuous, FS_SIM)
freqs_sampled, mag_sampled = compute_spectrum(x_sample, fs)

aliasing_result = check_aliasing(signal_freqs, fs)
alias_annotations = []
if aliasing_result['aliased_freqs']:
    for af in aliasing_result['aliased_freqs']:
        alias_annotations.append({'original': af['original'], 'alias': af['alias']})

x_reconstructed = reconstruct_signal(t_sample, x_sample, t_continuous, fs)
error_metrics = compute_reconstruction_error(x_continuous, x_reconstructed)

analysis = agent.analyze(
    signal_freqs=signal_freqs, fs=fs, aliasing_result=aliasing_result,
    noise_level=noise_level, reconstruction_error=error_metrics['mse'],
)
fmax = analysis.fmax
nyquist_freq = analysis.nyquist_freq
ratio = analysis.ratio
is_aliasing = analysis.is_aliasing

# ══════════════════════════════════════════════════════════════════════
# KPI 指标卡片行
# ══════════════════════════════════════════════════════════════════════

badge = get_sampling_status_badge(ratio, is_aliasing)
sc = get_status_color(ratio, is_aliasing)
dot_colors = {'#28a745': '#22c55e', '#ffc107': '#f59e0b', '#fd7e14': '#f97316',
              '#dc3545': '#ef4444', '#007bff': '#3b82f6'}
dot_c = dot_colors.get(sc, '#38bdf8')

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">最高频率</div>
        <div class="kpi-value">{fmax:.0f} <span class="kpi-unit">Hz</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">采样率</div>
        <div class="kpi-value">{fs:.0f} <span class="kpi-unit">Hz</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">采样比</div>
        <div class="kpi-value">{ratio:.2f}<span class="kpi-unit">×</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">奈奎斯特频率</div>
        <div class="kpi-value">{nyquist_freq:.0f} <span class="kpi-unit">Hz</span></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">重建 MSE</div>
        <div class="kpi-value">{error_metrics['mse']:.4f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">系统状态</div>
        <div class="kpi-value" style="color:{sc}; font-size:1.2rem;">
            <span class="status-dot" style="background:{dot_c}; color:{dot_c};"></span>{badge}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# Tab 分页
# ══════════════════════════════════════════════════════════════════════

t1, t2, t3, t4 = st.tabs([
    " 实时采样仿真",
    " 频谱分析实验室",
    " 信号重建实验",
    " DSP 智能分析",
])

# ── TAB 1: 实时采样仿真 ──
with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_time_domain(t_continuous, x_continuous, "原始连续信号"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(plot_sampling(t_continuous, x_continuous, t_sample, x_sample,
                                       f"采样过程  (fs={fs} Hz, {len(t_sample)} 点)"),
                        use_container_width=True)

# ── TAB 2: 频谱分析实验室 ──
with t2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_spectrum(
            freqs_orig, mag_orig,
            title="原始信号频谱（仿真采样率 5 kHz）",
            max_freq=min(MAX_DISPLAY_FREQ, FS_SIM / 2),
        ), use_container_width=True)
    with c2:
        st.plotly_chart(plot_spectrum(
            freqs_sampled, mag_sampled,
            title=f"采样后频谱  (fs={fs} Hz)",
            nyquist_freq=nyquist_freq,
            alias_annotations=alias_annotations,
            max_freq=min(MAX_DISPLAY_FREQ, max(fs, nyquist_freq * 2)),
            color='#6ee7b7',
        ), use_container_width=True)

# ── TAB 3: 信号重建实验 ──
with t3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_reconstruction(
            t_continuous, x_continuous, t_continuous, x_reconstructed,
            t_sample, x_sample,
            title="原始信号 vs sinc 重建",
        ), use_container_width=True)
    with c2:
        st.plotly_chart(plot_reconstruction_error(t_continuous, x_continuous, x_reconstructed),
                        use_container_width=True)

# ── TAB 4: DSP 智能分析 ──
with t4:
    cg, cs = st.columns([1, 2])

    with cg:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_sampling_gauge(ratio, is_aliasing), use_container_width=True)
        st.metric("重建 MSE", f"{error_metrics['mse']:.6f}")
        st.metric("重建 MAE", f"{error_metrics['mae']:.4f}")
        st.metric("最大误差", f"{error_metrics['max_error']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with cs:
        st.markdown("""
        <div class="agent-panel">
            <div class="agent-header">AI Sampling Agent &mdash; 分析报告</div>
        """, unsafe_allow_html=True)

        st.markdown("**系统分析**")
        for line in analysis.summary_lines:
            st.markdown(line)

        st.markdown("")
        st.markdown("**建议与优化方案**")
        for i, s in enumerate(analysis.suggestions, 1):
            if any(kw in s for kw in ["提高采样率", "混叠", "失真", "误差很大"]):
                icon = "⚠️"
            elif any(kw in s for kw in ["临界", "建议", "中等"]):
                icon = "⚡"
            else:
                icon = "✅"
            st.markdown(f"{i}. {icon} {s}")

        st.markdown('</div>', unsafe_allow_html=True)

    # 动态演示
    st.markdown("")
    st.markdown("##### 动态混叠演示")

    with st.expander("展开动态演示", expanded=False):
        st.markdown(
            "采样率从高到低自动扫描 —— "
            "观察混叠峰的出现与漂移、重建信号从贴合到崩溃。"
        )
        if st.button("▶  开始演示混叠过程", type="primary", use_container_width=True):
            fs_start = max(400, int(2.5 * fmax))
            fs_end = max(10, int(0.3 * fmax))
            fs_sweep = np.linspace(fs_start, fs_end, 60)

            status = st.empty()
            cs1, cs2 = st.columns(2)
            sp = cs1.empty()
            rp = cs2.empty()
            pb = st.progress(0)

            for i, fs_demo in enumerate(fs_sweep):
                t_s, x_s = sample_signal(t_continuous, x_continuous, fs_demo)
                f_s, m_s = compute_spectrum(x_s, fs_demo)
                al = check_aliasing(signal_freqs, fs_demo)
                anns = [{'original': a['original'], 'alias': a['alias']}
                        for a in al.get('aliased_freqs', [])]
                x_r = reconstruct_signal(t_s, x_s, t_continuous, fs_demo)

                if al['is_aliasing']:
                    desc = "、".join(f"{a['original']:.0f}→{a['alias']:.0f}Hz"
                                     for a in al['aliased_freqs'])
                    status.error(f"fs={fs_demo:.0f} Hz  |  fs/fmax={fs_demo/fmax:.2f}  |  混叠: {desc}")
                elif al['ratio'] < 2.2:
                    status.warning(f"fs={fs_demo:.0f} Hz  |  fs/fmax={fs_demo/fmax:.2f}  |  临界采样")
                else:
                    status.success(f"fs={fs_demo:.0f} Hz  |  fs/fmax={fs_demo/fmax:.2f}  |  正常")

                with sp.container():
                    st.plotly_chart(plot_spectrum(
                        f_s, m_s, title=f"采样后频谱  (fs={fs_demo:.0f} Hz)",
                        nyquist_freq=fs_demo / 2, alias_annotations=anns,
                        max_freq=min(MAX_DISPLAY_FREQ, max(fs_demo, 300)),
                        color='#6ee7b7',
                    ), use_container_width=True)

                with rp.container():
                    st.plotly_chart(plot_reconstruction(
                        t_continuous, x_continuous, t_continuous, x_r, t_s, x_s,
                        title=f"重建对比  (fs={fs_demo:.0f} Hz)",
                    ), use_container_width=True)

                pb.progress((i + 1) / len(fs_sweep))
                time.sleep(0.08)

            pb.empty()
            st.success("扫描完成，拖动上方 fs 滑块可手动探索。")

# ══════════════════════════════════════════════════════════════════════
# DSP 理论基础（可折叠）
# ══════════════════════════════════════════════════════════════════════

with st.expander(" DSP 理论基础参考", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **奈奎斯特采样定理**

        $$f_s \\ge 2 \\cdot f_{max}$$

        采样率必须 ≥ 信号最高频率的 2 倍，才能无失真重建。
        """)
        st.markdown("""
        **混叠频率公式**

        $$f_{alias} = |f_{signal} - k \\cdot f_s|$$

        取 k 使得结果落在 $[0, f_s/2]$ 区间。
        """)
    with c2:
        st.markdown("""
        **sinc 插值重建**

        $$x(t) = \\sum_{n} x[n] \\cdot \\text{sinc}\\left(\\frac{t - nT_s}{T_s}\\right)$$

        $$\\text{sinc}(x) = \\frac{\\sin(\\pi x)}{\\pi x}$$
        """)
        st.markdown("""
        **采样状态分类**

        | fs/fmax | 状态 |
        |---|---|
        | < 2 | 欠采样 / 混叠 |
        | ≈ 2 | 临界采样 |
        | 2~5 | 正常采样 |
        | > 5 | 过采样 |
        """)

# ══════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "AliasLab v1.0  |  数字信号处理课程大作业  |  "
    "采样-混叠-重建智能仿真系统  |  DSP Intelligent Experiment Platform"
)
