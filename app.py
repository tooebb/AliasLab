"""
AliasLab — 采样混叠与重建智能仿真系统
========================================
面向数字信号处理教学的采样-混叠-重建全过程智能分析平台

主应用入口 — 仅负责 UI 布局与模块调度，
所有 DSP 逻辑分布在 core/、agent/、utils/ 中。
"""

import streamlit as st
import numpy as np
import time

# ─── DSP 核心模块 ───
from core.signal_generator import generate_time_array, generate_signal
from core.sampling import sample_signal
from core.aliasing import check_aliasing
from core.fft_analysis import compute_spectrum
from core.reconstruction import reconstruct_signal, compute_reconstruction_error

# ─── 智能 Agent ───
from agent.sampling_agent import SamplingAgent

# ─── 可视化 & 工具 ───
from utils.plotting import (
    plot_time_domain,
    plot_sampling,
    plot_spectrum,
    plot_reconstruction,
    plot_reconstruction_error,
    plot_sampling_gauge,
)
from utils.helpers import get_sampling_status_badge, get_status_color

# ══════════════════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AliasLab — 采样混叠与重建智能仿真",
    page_icon="",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════════════════

FS_SIM = 5000        # 仿真采样率 (Hz)，固定高值模拟连续信号
MAX_DISPLAY_FREQ = 300  # 频谱图最大显示频率 (Hz)

# ══════════════════════════════════════════════════════════════════════
# 初始化 Agent（单例）
# ══════════════════════════════════════════════════════════════════════

if 'agent' not in st.session_state:
    st.session_state.agent = SamplingAgent()

agent = st.session_state.agent

# ══════════════════════════════════════════════════════════════════════
# 页面标题
# ══════════════════════════════════════════════════════════════════════

st.title("AliasLab — 采样混叠与重建智能仿真系统")
st.caption("面向数字信号处理教学的采样-混叠-重建全过程智能分析平台")

# ══════════════════════════════════════════════════════════════════════
# 侧边栏：参数控制
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("信号参数")

    signal_type = st.selectbox(
        "信号类型",
        ["单频正弦波", "双频正弦波", "方波", "含噪正弦波"],
    )

    duration = st.slider("信号时长 (秒)", 0.1, 2.0, 0.5, 0.05,
                         help="信号的时间长度，越长 FFT 频率分辨率越高")

    # ── 根据类型收集参数 ──
    noise_level = 0.0

    if signal_type == "单频正弦波":
        freq1 = st.slider("信号频率 f (Hz)", 1, 300, 50, 1)
        amp1 = st.slider("振幅 A", 0.1, 5.0, 1.0, 0.1)
        freq2, amp2 = None, None

    elif signal_type == "双频正弦波":
        c1, c2 = st.columns(2)
        with c1:
            freq1 = st.slider("频率 f₁ (Hz)", 1, 300, 40, 1)
            amp1 = st.slider("振幅 A₁", 0.1, 5.0, 1.0, 0.1)
        with c2:
            freq2 = st.slider("频率 f₂ (Hz)", 1, 300, 150, 1)
            amp2 = st.slider("振幅 A₂", 0.1, 5.0, 0.5, 0.1)
        st.caption("提示: 设置 f₂ > fs/2 可观察混叠现象")

    elif signal_type == "方波":
        freq1 = st.slider("基频 f (Hz)", 1, 100, 10, 1)
        amp1 = st.slider("振幅 A", 0.1, 5.0, 1.0, 0.1)
        freq2, amp2 = None, None
        st.info("方波包含奇次谐波 (3f, 5f, 7f, 9f)，"
                "实际带宽远高于基频。需以最高谐波频率计算奈奎斯特率。")

    elif signal_type == "含噪正弦波":
        freq1 = st.slider("信号频率 f (Hz)", 1, 300, 50, 1)
        amp1 = st.slider("振幅 A", 0.1, 5.0, 1.0, 0.1)
        noise_level = st.slider("噪声强度 σ", 0.0, 1.0, 0.2, 0.05,
                                help="高斯白噪声的标准差")
        freq2, amp2 = None, None

    st.markdown("---")
    st.header("采样参数")

    fs = st.slider("采样率 fs (Hz)", 1, 600, 200, 1,
                   help="ADC 采样频率。低于 2×fmax 时发生混叠")

    st.markdown("---")

    # ── 实时状态指示 ──
    fmax_preview = freq1
    if signal_type == "双频正弦波" and freq2:
        fmax_preview = max(freq1, freq2)
    elif signal_type == "方波":
        fmax_preview = freq1 * 9  # 考虑至 9 次谐波

    ratio_preview = fs / fmax_preview if fmax_preview > 0 else float('inf')
    is_aliasing_preview = fs < 2 * fmax_preview
    badge = get_sampling_status_badge(ratio_preview, is_aliasing_preview)
    color = get_status_color(ratio_preview, is_aliasing_preview)

    st.markdown(
        f"<h3 style='text-align:center; color:{color};'>状态: {badge}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    | 参数 | 数值 |
    |------|------|
    | 最高频率 fmax | **{fmax_preview} Hz** |
    | 所需奈奎斯特率 | **{2 * fmax_preview} Hz** |
    | 采样比 fs/fmax | **{ratio_preview:.2f}** |
    | 奈奎斯特频率 fs/2 | **{fs / 2:.1f} Hz** |
    """)

# ══════════════════════════════════════════════════════════════════════
# 信号生成
# ══════════════════════════════════════════════════════════════════════

t_continuous = generate_time_array(duration, FS_SIM)

if signal_type == "单频正弦波":
    x_continuous, signal_freqs = generate_signal(
        t_continuous, 'single_sine', freq=freq1, amp=amp1)
elif signal_type == "双频正弦波":
    x_continuous, signal_freqs = generate_signal(
        t_continuous, 'dual_sine',
        freq1=freq1, amp1=amp1, freq2=freq2, amp2=amp2)
elif signal_type == "方波":
    x_continuous, signal_freqs = generate_signal(
        t_continuous, 'square', freq=freq1, amp=amp1)
elif signal_type == "含噪正弦波":
    x_continuous, signal_freqs = generate_signal(
        t_continuous, 'noisy_sine',
        freq=freq1, amp=amp1, noise_level=noise_level)

# ══════════════════════════════════════════════════════════════════════
# 采样
# ══════════════════════════════════════════════════════════════════════

t_sample, x_sample = sample_signal(t_continuous, x_continuous, fs)

# ══════════════════════════════════════════════════════════════════════
# FFT 频谱分析
# ══════════════════════════════════════════════════════════════════════

freqs_orig, mag_orig = compute_spectrum(x_continuous, FS_SIM)
freqs_sampled, mag_sampled = compute_spectrum(x_sample, fs)

# ══════════════════════════════════════════════════════════════════════
# 混叠分析
# ══════════════════════════════════════════════════════════════════════

aliasing_result = check_aliasing(signal_freqs, fs)

# 构建混叠标注数据（用于频谱图）
alias_annotations = []
if aliasing_result['aliased_freqs']:
    for af in aliasing_result['aliased_freqs']:
        alias_annotations.append({
            'original': af['original'],
            'alias': af['alias'],
        })

# ══════════════════════════════════════════════════════════════════════
# sinc 信号重建
# ══════════════════════════════════════════════════════════════════════

x_reconstructed = reconstruct_signal(t_sample, x_sample, t_continuous, fs)
error_metrics = compute_reconstruction_error(x_continuous, x_reconstructed)

# ══════════════════════════════════════════════════════════════════════
# Agent 智能分析
# ══════════════════════════════════════════════════════════════════════

analysis = agent.analyze(
    signal_freqs=signal_freqs,
    fs=fs,
    aliasing_result=aliasing_result,
    noise_level=noise_level,
    reconstruction_error=error_metrics['mse'],
)

fmax = analysis.fmax
nyquist_freq = analysis.nyquist_freq
ratio = analysis.ratio
is_aliasing = analysis.is_aliasing

# ══════════════════════════════════════════════════════════════════════
# 主区域：图表展示
# ══════════════════════════════════════════════════════════════════════

# ── Row 1: 时域波形 + 采样过程 ──
st.markdown("### 时域分析")
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        plot_time_domain(t_continuous, x_continuous, "原始连续信号"),
        use_container_width=True,
    )

with col_right:
    st.plotly_chart(
        plot_sampling(t_continuous, x_continuous,
                       t_sample, x_sample,
                       f"采样过程 (fs = {fs} Hz, {len(t_sample)} 个采样点)"),
        use_container_width=True,
    )

# ── Row 2: 频谱对比 ──
st.markdown("---")
st.markdown("### 频谱分析")

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        plot_spectrum(
            freqs_orig, mag_orig,
            title="原始信号频谱（高分辨率仿真）",
            nyquist_freq=None,
            max_freq=min(MAX_DISPLAY_FREQ, FS_SIM / 2),
        ),
        use_container_width=True,
    )

with col_right:
    st.plotly_chart(
        plot_spectrum(
            freqs_sampled, mag_sampled,
            title=f"采样后频谱 (fs = {fs} Hz)",
            nyquist_freq=nyquist_freq,
            alias_annotations=alias_annotations,
            max_freq=min(MAX_DISPLAY_FREQ, max(fs, nyquist_freq * 2)),
            color='#1f77b4',
        ),
        use_container_width=True,
    )

# ── Row 3: 信号重建 ──
st.markdown("---")
st.markdown("### 信号重建 (sinc 插值)")

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(
        plot_reconstruction(
            t_continuous, x_continuous,
            t_continuous, x_reconstructed,
            t_sample, x_sample,
            title="原始信号 vs sinc 重建信号",
        ),
        use_container_width=True,
    )

with col_right:
    st.plotly_chart(
        plot_reconstruction_error(t_continuous, x_continuous, x_reconstructed),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════
# 智能分析报告
# ══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("DSP 智能分析报告")

# 仪表盘 + 摘要并排
col_gauge, col_summary = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(
        plot_sampling_gauge(ratio, is_aliasing),
        use_container_width=True,
    )
    # 误差指标
    st.metric("重建 MSE", f"{error_metrics['mse']:.6f}")
    st.metric("重建 MAE", f"{error_metrics['mae']:.4f}")
    st.metric("最大误差", f"{error_metrics['max_error']:.4f}")

with col_summary:
    st.markdown("#### 系统分析")
    for line in analysis.summary_lines:
        st.markdown(line)

    st.markdown("#### 建议与优化方案")
    for i, suggestion in enumerate(analysis.suggestions, 1):
        if any(kw in suggestion for kw in ["提高采样率", "混叠", "失真", "误差很大"]):
            icon = "⚠️"
        elif any(kw in suggestion for kw in ["临界", "建议", "中等"]):
            icon = "⚡"
        else:
            icon = "✅"
        st.markdown(f"{i}. {icon} {suggestion}")

# ══════════════════════════════════════════════════════════════════════
# 动态混叠演示
# ══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("动态混叠演示")

st.markdown("""
点击下方按钮，系统将自动扫描采样率 **从高到低**，
你可以实时观察混叠峰如何出现并沿频率轴漂移，
以及重建信号如何从完美逐渐崩溃。
""")

if st.button("▶  开始演示混叠过程", type="primary", use_container_width=True):
    # ── 计算扫频范围 ──
    fs_start = max(400, int(2.5 * fmax))
    fs_end = max(10, int(0.3 * fmax))
    fs_sweep = np.linspace(fs_start, fs_end, 60)

    status_placeholder = st.empty()
    col_spec, col_recon = st.columns(2)
    spec_placeholder = col_spec.empty()
    recon_placeholder = col_recon.empty()
    progress_bar = st.progress(0)

    for i, fs_demo in enumerate(fs_sweep):
        # 采样
        t_s, x_s = sample_signal(t_continuous, x_continuous, fs_demo)

        # FFT
        f_s, m_s = compute_spectrum(x_s, fs_demo)

        # 混叠分析
        al_result = check_aliasing(signal_freqs, fs_demo)
        anns = [{'original': af['original'], 'alias': af['alias']}
                for af in al_result.get('aliased_freqs', [])]

        # sinc 重建
        x_r = reconstruct_signal(t_s, x_s, t_continuous, fs_demo)

        # 状态条
        if al_result['is_aliasing']:
            alias_desc = "、".join(
                f"{a['original']:.0f}→{a['alias']:.0f}Hz"
                for a in al_result['aliased_freqs']
            )
            status_placeholder.error(
                f"fs = **{fs_demo:.0f} Hz** | "
                f"fs/fmax = **{fs_demo/fmax:.2f}** | "
                f"混叠: {alias_desc}"
            )
        elif al_result['ratio'] < 2.2:
            status_placeholder.warning(
                f"fs = **{fs_demo:.0f} Hz** | "
                f"fs/fmax = **{fs_demo/fmax:.2f}** | "
                f"临界采样"
            )
        else:
            status_placeholder.success(
                f"fs = **{fs_demo:.0f} Hz** | "
                f"fs/fmax = **{fs_demo/fmax:.2f}** | "
                f"正常采样，无混叠"
            )

        # 更新频谱图
        with spec_placeholder.container():
            st.plotly_chart(
                plot_spectrum(
                    f_s, m_s,
                    title=f"采样后频谱 (fs={fs_demo:.0f} Hz)",
                    nyquist_freq=fs_demo / 2,
                    alias_annotations=anns,
                    max_freq=min(MAX_DISPLAY_FREQ, max(fs_demo, 300)),
                    color='#1f77b4',
                ),
                use_container_width=True,
            )

        # 更新重建对比图
        with recon_placeholder.container():
            st.plotly_chart(
                plot_reconstruction(
                    t_continuous, x_continuous,
                    t_continuous, x_r,
                    t_s, x_s,
                    title=f"重建对比 (fs={fs_demo:.0f} Hz)",
                ),
                use_container_width=True,
            )

        progress_bar.progress((i + 1) / len(fs_sweep))
        time.sleep(0.08)

    progress_bar.empty()
    st.success("演示完成！fs 已从高到低扫描完毕，你可以拖动上方滑块手动探索。")

# ══════════════════════════════════════════════════════════════════════
# 页脚
# ══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "AliasLab v1.0 | 数字信号处理课程大作业 | "
    "采样-混叠-重建智能仿真系统 | "
    "基于规则推理的 DSP 实验助手"
)
